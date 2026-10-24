from ikaggle.AAA.config.feature_config import get_gene_length, FEATURE_CONFIGS  # Add import

def calculate_sample_size(gen, test_mode):
    """Calculate dynamic sample size based on generation number"""
    if test_mode:
        return 1000  # Small sample for testing
        
    # Start with 10k and increase up to 50k
    base_size = 10000
    max_size = 50000
    size_step = 5000
    
    # Increase sample size every 5 generations
    size = base_size + (gen // 5) * size_step
    return min(size, max_size)

def get_fitness_stats(population):
    print('in get_fitness_stats')
    fitnesses = []
    for ind in population:
        # Check if fitness.values exists and has at least one element.
        if hasattr(ind, 'fitness') and ind.fitness.values and len(ind.fitness.values) > 0:
            fitnesses.append(ind.fitness.values[0])
        else:
            # Use a default high value (or float('inf')) if not evaluated.
            fitnesses.append(float('inf'))
    avg = sum(fitnesses) / len(fitnesses) if fitnesses else 0
    std = (sum((x - avg) ** 2 for x in fitnesses) / len(fitnesses)) ** 0.5 if fitnesses else 0
    print(avg, std)
    return {
        'gen': 0,
        'nevals': len(population),
        'avg': avg,
        'std': std,
        'min': min(fitnesses) if fitnesses else 0,
        'max': max(fitnesses) if fitnesses else 0
    }

def count_active_features(individual, feature_columns):
    """Count active features in an individual"""
    active_count = 0
    current_pos = 0
    
    # Access genome from Individual object
    genome = individual.genome
    
    # Count base features
    for feature_name in feature_columns:
        if genome[current_pos]:  # Changed from individual[current_pos]
            active_count += 1
        current_pos += 1
        # Skip variable length genes if needed
        if current_pos < len(genome):  # Changed from len(individual)
            if feature_name in FEATURE_CONFIGS and FEATURE_CONFIGS[feature_name]['type'] == 'variable':
                current_pos += get_gene_length(feature_name)
    
    # Count active pairs and triplets
    for i in range(current_pos, len(genome)):  # Changed from len(individual)
        if genome[i]:  # Changed from individual[i]
            active_count += 1
    
    return active_count

# Define a maximum number of generations for the evolution loop.
max_generations = 50  # Increased from default
