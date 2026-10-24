import os
import sys
import logging
from ikaggle.AAA.config.paths import (
    CACHE_PATHS, DATA_PATHS, USER_CUDA_DIR, 
    CONDA_ENV, USER_CUDA_NVVM, CUDA_PATHS
)
# Add logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ikaggle')

import tensorflow as tf
import datetime
import numpy as np
import pandas as pd

# Setup CUDA paths before importing tensorflow
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={USER_CUDA_DIR}'
os.environ['CUDA_DIR'] = USER_CUDA_DIR
os.environ['XLA_FLAGS'] = (
    f'--xla_gpu_cuda_data_dir={os.path.join(CONDA_ENV, "lib")} '
    f'{os.environ.get("XLA_FLAGS", "")}'
)

from ikaggle.AAA.utils.data_loading import DataLoader
from ikaggle.AAA.utils.gpu_manager import GPUManager
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS, get_gene_length  # Add this import
from ikaggle.AAA.utils.error_handling import log_error  # Add this import

# Suppress TensorFlow NUMA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Updated imports with full package paths
from ikaggle.AAA.utils.data_loading import DataLoader, GENOME_TEMPLATE, TOTAL_GENE_LENGTH
from ikaggle.AAA.experiments.genetic_ops import setup_toolbox
from ikaggle.AAA.experiments.model_trainer import fitness_function, parallel_fitness_evaluation, PARALLEL_INDIVIDUALS
from ikaggle.AAA.experiments.evolution_monitor import setup_tensorboard, log_evolution_stats
from ikaggle.AAA.experiments.evolution_utils import calculate_sample_size, get_fitness_stats, count_active_features, max_generations
from ikaggle.AAA.experiments.submission_handler import create_submission
from ikaggle.AAA.utils.persistence import save_generation, load_generation

from deap import tools, algorithms  # Needed for evolutionary operations

# Dummy implementations for missing functions for testing purposes
def configure_tensorflow():
    """Configure TensorFlow with proper CUDA paths"""
    try:
        # Get GPUs first
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # First set memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Then configure memory limit
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 13)]
                )
                
                print(f"\nGPU Configuration:")
                print(f"Found {len(gpus)} GPU(s)")
                print(f"Memory Growth: Enabled")
                print(f"Memory Limit: 13GB")
                
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
    except Exception as e:
        print(f"Error configuring TensorFlow: {e}")

def setup_gpu():
    return None, True

def cleanup():
    pass

# Define paths
GRAPH_DUMP_DIR = os.path.join(os.path.dirname(__file__), 'tf_graphs')
BEST_MODEL_PATH = os.path.join(CACHE_PATHS['models'], 'best_model.keras')
STATS_PATH = os.path.join(CACHE_PATHS['models'], 'evolution_stats.csv')
CHECKPOINT_PATH = os.path.join(CACHE_PATHS['models'], 'evolution_checkpoint.pkl')
POPULATION_PATH = os.path.join(CACHE_PATHS['models'], 'last_population.pkl')
GENERATION_DIR = os.path.join(CACHE_PATHS['models'], 'generations')

# Create required directories
for directory in [GRAPH_DUMP_DIR, GENERATION_DIR]:
    os.makedirs(directory, exist_ok=True)

def analyze_feature_activation(population, X_full):
    """Analyze which features are being used across the population"""
    n_individuals = len(population)
    n_base_features = len(X_full.columns)
    
    # Track activation counts
    feature_counts = {col: 0 for col in X_full.columns}
    pair_counts = 0
    triplet_counts = 0
    
    for ind in population:
        # Access genome from individual
        genome = ind.genome
        
        # Base features are at the start of genome
        current_pos = 0
        for feature in X_full.columns:
            if genome[current_pos]:  # Check activation bit using genome
                feature_counts[feature] += 1
            current_pos += 1  # Move past activation bit
            if feature in FEATURE_CONFIGS and FEATURE_CONFIGS[feature]['type'] == 'variable':
                gene_length = get_gene_length(feature)
                current_pos += gene_length
        
        # Count active combinations
        for i in range(200):  # Number of possible pairs
            if current_pos >= len(genome):
                break
            if genome[current_pos]:  # Pair activation bit using genome
                pair_counts += 1
            current_pos += 21
            
        for i in range(50):  # Number of possible triplets
            if current_pos >= len(genome):
                break
            if genome[current_pos]:  # Triplet activation bit using genome
                triplet_counts += 1
            current_pos += 31
    
    # Calculate statistics
    avg_base_features = sum(feature_counts.values()) / n_individuals
    avg_pairs = pair_counts / n_individuals
    avg_triplets = triplet_counts / n_individuals
    
    return feature_counts, avg_base_features, avg_pairs, avg_triplets

def log_error(error, message="Error"):
    """Helper function to format error messages"""
    import traceback
    tb = traceback.format_exception(type(error), error, error.__traceback__)
    return f"{message}: {str(error)}\n{''.join(tb)}"

def main(X_full, y_full, verbose=False, test_mode=False):
    """Main evolution loop with modular components"""
    # Input validation
    if X_full is None or y_full is None:
        raise ValueError("Invalid input data: X_full and y_full must not be None")
    
    if len(X_full) == 0 or len(y_full) == 0:
        raise ValueError("Empty input data: X_full and y_full must not be empty")
    
    print(f"Input data shapes - X: {X_full.shape}, y: {len(y_full)}")
    
    # Setup with optimized population size, feature columns and configs
    toolbox = setup_toolbox(
        X_columns=X_full.columns,
        feature_configs=FEATURE_CONFIGS  # Pass feature configs explicitly
    )
    writer = setup_tensorboard(CACHE_PATHS['models'])
    
    # Increase population size to better utilize GPU
    min_pop_size = 20  # Force minimum population to 20
    population_size = 20  # Fixed population size
    
    # Adjust elite size and tournament size for larger population
    ELITE_SIZE = 5  # Keep top 25%
    TOURNAMENT_SIZE = 6  # 30% of population
    
    # Create population with increased size
    population = toolbox.population(n=population_size)
    
    # Add individual tracking
    for i, ind in enumerate(population):
        ind.id = f"G0_I{i}"  # Generation 0, Individual i
        ind.birth_gen = 0
    
    print(f"\n=== Starting Generation 0 ===")
    print(f"Population size: {len(population)}")
    
    # Initial evaluation with larger sample
    initial_sample_size = min(10000, len(y_full))
    indices = np.random.choice(len(y_full), size=initial_sample_size, replace=False)
    X_sample = X_full.iloc[indices]
    y_sample = y_full.iloc[indices]
    
    if 'id' in X_sample.columns:
        X_sample = X_sample.drop(columns=['id'])
    
    # Ensure initial population is properly evaluated
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        try:
            fit = fitness_function(ind, X_sample, y_sample, verbose=(verbose > 0))
            if fit is None or fit[0] is None:
                # Handle failed fitness calculation
                fit = (float('inf'),)
            if not isinstance(fit, tuple):
                # Ensure we have a tuple
                fit = (float(fit),)
            ind.fitness.values = fit
        except Exception as e:
            print(f"Error calculating fitness: {str(e)}")
            ind.fitness.values = (float('inf'),)

    # Initialize hall of fame
    hof = tools.HallOfFame(10)
    hof.update(population)
    
    if verbose:
        print(f"Initial population created: {len(population)} individuals")
        print("Hall of Fame:")
        for i, ind in enumerate(hof):
            print(f"HoF #{i+1} - Fitness: {ind.fitness.values[0]:.4f}")
    
    # Log initial stats
    stats = get_fitness_stats(population)
    log_evolution_stats(writer, 0, stats['min'], stats['avg'],
                       count_active_features(hof[0], X_full.columns), 
                       calculate_sample_size(0, test_mode), 0)

    try:
        gen = 0
        stagnant_generations = 0
        best_fitness = float('inf')
        best_individual = None
        min_generations = 20  # Increased minimum generations
        max_stagnant = 10    # Increased stagnation tolerance
        previous_population = []
        
        # Save initial population
        save_generation(population, 0, GENERATION_DIR)
        
        while ((stagnant_generations < max_stagnant or gen < min_generations) 
               and gen < max_generations):
            gen += 1
            previous_population = list(population)  # Save current population
            
            # Save previous generation before modifications
            save_dir = save_generation(previous_population, gen-1, GENERATION_DIR)
            if verbose:
                print(f"\nSaved generation {gen-1} to {save_dir}")
            
            # Calculate sample size first
            sample_size = calculate_sample_size(gen, test_mode)
            print(f"\n=== Generation {gen} ===")
            print(f"Sample size: {sample_size}")
            print(f"Stagnant generations: {stagnant_generations}")
            
            # Get sample data
            indices = np.random.choice(len(y_full), size=sample_size, replace=False)
            X_sample = X_full.iloc[indices]
            y_sample = y_full.iloc[indices]
            
            if 'id' in X_sample.columns:
                X_sample = X_sample.drop(columns=['id'])
            
            # Select elite individuals
            elite = tools.selBest(population, ELITE_SIZE)
            
            # Create offspring
            offspring = algorithms.varAnd(
                population, 
                toolbox,
                cxpb=0.7,
                mutpb=0.3
            )
            
            # Create offspring with tracking
            for i, ind in enumerate(offspring):
                if not hasattr(ind, 'id'):  # New individual from crossover/mutation
                    ind.id = f"G{gen}_I{i}"
                    ind.birth_gen = gen
            
            # Evaluate offspring in parallel batches
            with tf.device('/GPU:0'):
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                batch_size = min(PARALLEL_INDIVIDUALS, len(invalid_ind))
                
                for i in range(0, len(invalid_ind), batch_size):
                    batch_inds = invalid_ind[i:i + batch_size]
                    results = parallel_fitness_evaluation(batch_inds, X_sample, y_sample)
                    
                    for ind, (fitness, _) in zip(batch_inds, results):
                        ind.fitness.values = (fitness,)

            # Before selecting next generation, mark removed individuals
            next_gen_ids = {ind.id for ind in elite + offspring}
            for ind in previous_population:
                if ind.id not in next_gen_ids:
                    ind.mark_removed(gen, "not selected")
            
            # Select next generation
            population = elite + tools.selBest(offspring, len(population) - ELITE_SIZE)
            
            # Log population changes
            if verbose:
                removed_count = sum(1 for ind in previous_population if ind.removal_gen == gen)
                print(f"\nPopulation changes in generation {gen}:")
                print(f"Removed individuals: {removed_count}")
                print(f"Average lifetime: {np.mean([ind.lifetime_generations for ind in previous_population if ind.removal_gen == gen]):.1f} generations")
            
            # Update hall of fame
            hof.update(population)
            
            # Update statistics and check for improvement
            stats = get_fitness_stats(population)
            current_best = stats['min']
            
            if current_best < best_fitness:
                best_fitness = current_best
                stagnant_generations = 0
                best_individual = tools.selBest(population, 1)[0]
                
                print(f"\nNew best fitness: {best_fitness:.4f}")
                print("Creating submission...")
                
                try:
                    X_test = data_loader.load_dataset('test')
                    if X_test is not None:
                        _, predictions = fitness_function(
                            best_individual, 
                            X_test, 
                            pd.Series([0] * len(X_test)),
                            return_predictions=True,
                            verbose=True
                        )
                        
                        if predictions is not None:
                            # Create submission will now exit on failure
                            submission_path = create_submission(
                                best_individual,
                                predictions,
                                X_test.index,
                                gen,
                                individual_id=best_individual.id
                            )
                            print(f"Submission saved and verified: {submission_path}")
                        else:
                            raise ValueError("No predictions generated")
                    else:
                        raise ValueError("Could not load test data")
                        
                except Exception as e:
                    print(f"Submission creation failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    sys.exit(1)  # Stop program on any submission-related failure
                
            else:
                stagnant_generations += 1
            
            # Log progress
            if verbose:
                print(f"\nGeneration {gen}:")
                print(f"Population size: {len(population)}")
                print(f"Best fitness: {best_fitness:.4f}")
                print(f"Stagnant generations: {stagnant_generations}")
            
            log_evolution_stats(writer, gen, stats['min'], stats['avg'],
                              count_active_features(hof[0], X_full.columns), 
                              sample_size, stagnant_generations)
            
            # Add activation analysis every 10 generations
            if gen % 10 == 0:
                feature_counts, avg_base, avg_pairs, avg_triplets = analyze_feature_activation(population, X_full)
                log_evolution_stats(writer, gen, stats['min'], stats['avg'],
                                  avg_base + avg_pairs + avg_triplets,
                                  calculate_sample_size(gen, test_mode), 
                                  stagnant_generations)
                
            # Log current best individual info
            best_ind = tools.selBest(population, 1)[0]
            print(f"Best individual: {best_ind.id} (born gen {best_ind.birth_gen})")
            print(f"Fitness: {best_ind.fitness.values[0]:.4f}")
    except KeyboardInterrupt:
        # Save final state before interruption
        save_generation(population, gen, GENERATION_DIR)
        logger.info(f"\nSaved interrupted generation {gen}")
        logger.info("\nEvolution interrupted by user")
        
        # Mark remaining individuals as removed
        for ind in population:
            if ind.removal_gen is None:
                ind.mark_removed(gen, "evolution interrupted")
    
    # Create final submission with best individual
    if best_individual is not None:
        try:
            X_test = data_loader.load_dataset('test')
            if X_test is not None:
                _, predictions = fitness_function(
                    best_individual, 
                    X_test, 
                    pd.Series([0] * len(X_test)),
                    return_predictions=True,
                    verbose=True
                )
                
                if predictions is not None:
                    submission_path = create_submission(
                        best_individual,
                        predictions,
                        X_test.index,
                        gen,
                        individual_id=best_individual.id
                    )
                    print(f"Final submission saved and verified: {submission_path}")
                else:
                    raise ValueError("No predictions generated")
            else:
                raise ValueError("Could not load test data")
                
        except Exception as e:
            print(f"Final submission creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)  # Stop program on any submission-related failure
    
    return population, stats, list(hof)

if __name__ == "__main__":
    try:
        # Configure GPU first
        configure_tensorflow()
        gpu_device, GPU_AVAILABLE = setup_gpu()
        
        if not GPU_AVAILABLE:
            print("WARNING: No GPU available, performance will be limited")
        else:
            print(f"Using GPU: {gpu_device}")
        
        # Create data loader and verify paths
        data_loader = DataLoader(DATA_PATHS)
        
        # Load both train and test data at start
        X_full, y_full = data_loader.load_dataset('train')
        X_test, _ = data_loader.load_dataset('test')
        
        if X_full is None or y_full is None or X_test is None:
            raise ValueError("Failed to load required datasets")
            
        print(f"Loaded data shapes:")
        print(f"Train: {X_full.shape}")
        print(f"Test: {X_test.shape}")
        
        # Run evolution with GPU forced
        with tf.device('/GPU:0'):
            pop, stats, hof = main(X_full, y_full, verbose=False, test_mode=False)
            
            # Create final submissions for top individuals
            print("\nCreating submissions for top individuals...")
            for i, ind in enumerate(hof[:3]):
                try:
                    _, predictions = fitness_function(
                        ind, X_test, pd.Series([0] * len(X_test)),
                        return_predictions=True
                    )
                    if predictions is not None:
                        create_submission(
                            ind,
                            predictions,
                            X_test.index,
                            gen=-1,  # Use -1 to indicate final submission
                            individual_id=f"HOF_{i+1}"
                        )
                except Exception as e:
                    print(f"Failed to create submission for HOF #{i+1}: {str(e)}")
                    
    except Exception as e:
        logger.error(log_error(e, "Main execution error"))
        sys.exit(1)
    finally:
        cleanup()
