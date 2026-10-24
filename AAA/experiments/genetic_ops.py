import random
from deap import base, creator, tools
from ikaggle.AAA.utils.data_loading import GENOME_TEMPLATE, TOTAL_GENE_LENGTH
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS, get_gene_length  # Add this import
from ikaggle.AAA.models.individual import Individual
import numpy as np

# Key features to always activate
# Update core features with their importance from statistical analysis
CORE_FEATURES = {
    'AnnualIncome': 0,       # Most important predictor
    'HealthScore': 0,        # Strong correlation
    'PreviousClaims': 0,     # Important for risk assessment
    'Age': 0,               # Base demographic
    'NumberofDependents': 0, # Key factor
    'VehicleAge': 0,        # Risk factor
    'CreditScore': 0        # Financial indicator
}

# Priority feature combinations (feature1, feature2, operation)
# Operation codes: 0=multiply, 1=add, 2=subtract, 3=divide
# Priority feature combinations updated based on statistical significance
PRIORITY_COMBINATIONS = [
    ('NumberofDependents', 'PreviousClaims', 0),  # p=0.010
    ('HealthScore', 'PreviousClaims', 0),         # p=0.019
    ('Age', 'HealthScore', 0),                    # p=0.021
    ('Age', 'PreviousClaims', 0),                 # p=0.124
    ('AnnualIncome', 'NumberofDependents', 0),    # p=0.633
]

def custom_crossover(ind1, ind2):
    """Custom crossover that preserves existing combinations"""
    # Record parentage
    child1 = Individual(genome=list(ind1.genome), parent_ids=[ind1.id])
    child2 = Individual(genome=list(ind2.genome), parent_ids=[ind2.id])
    
    # Perform crossover
    if len(child1.genome) != len(child2.genome):
        raise ValueError(f"Individuals have different lengths: {len(child1.genome)} vs {len(child2.genome)}")
    
    # Get actual length of individual for safe indexing
    ind_length = len(child1.genome)
    
    # Calculate positions safely
    try:
        base_end = next(i for i, x in enumerate(GENOME_TEMPLATE) if x[0].startswith('combo'))
    except (StopIteration, IndexError):
        base_end = ind_length // 3  # Fallback if GENOME_TEMPLATE is incorrect
    
    # Handle base features
    for i in range(0, min(base_end, ind_length), 2):
        if random.random() < 0.5:
            child1.genome[i], child2.genome[i] = child2.genome[i], child1.genome[i]
            if i + 1 < ind_length:
                child1.genome[i+1], child2.genome[i+1] = child2.genome[i+1], child1.genome[i+1]
    
    # Handle pair combinations with safe indexing
    pair_start = base_end
    pair_length = 5  # length of each pair combination
    num_pairs = (ind_length - pair_start) // (pair_length + 7)  # calculate max possible pairs
    
    for i in range(num_pairs):
        pos = pair_start + i * pair_length
        if pos + pair_length > ind_length:
            break
        
        if (child1.genome[pos] or child2.genome[pos]) and random.random() < 0.3:
            for j in range(pair_length):
                if pos + j < ind_length:
                    child1.genome[pos + j], child2.genome[pos + j] = child2.genome[pos + j], child1.genome[pos + j]
    
    # Record crossover
    child1.record_crossover(ind2.id)
    child2.record_crossover(ind1.id)
    
    return child1, child2

def custom_mutation(individual, indpb):
    """Custom mutation that heavily favors base features"""
    # Create new individual with mutation tracking
    child = Individual(genome=list(individual.genome), 
                      parent_ids=[individual.id],
                      birth_gen=individual.birth_gen + 1)
    
    ind_length = len(child.genome)
    
    # Calculate safe boundaries
    try:
        base_end = next(i for i, x in enumerate(GENOME_TEMPLATE) if x[0].startswith('combo'))
    except (StopIteration, IndexError):
        base_end = ind_length // 3  # Fallback if GENOME_TEMPLATE is incorrect
    
    # Base features - normal mutation rate
    for i in range(min(base_end, ind_length)):
        if random.random() < indpb:
            child.genome[i] = 1 - child.genome[i]
    
    # Pair combinations with safe indexing
    pair_start = base_end
    pair_length = 5  # length of each pair combination
    num_pairs = (ind_length - pair_start) // pair_length  # calculate max possible pairs
    
    for i in range(num_pairs):
        pos = pair_start + i * pair_length
        if pos >= ind_length:
            break
            
        # Mutation logic for pair
        if random.random() < 0.001:  # 0.1% chance to activate
            child.genome[pos] = 1
        elif random.random() < 0.01:  # 1% chance to deactivate
            child.genome[pos] = 0
        
        # Mutate parameters if active
        if child.genome[pos]:
            for j in range(1, min(pair_length, ind_length - pos)):
                if random.random() < indpb * 0.1:
                    child.genome[pos + j] = 1 - child.genome[pos + j]
    
    # Record mutation
    child.record_mutation()
    
    return child,

def setup_toolbox(X_columns=None, feature_configs=None):
    """Setup DEAP toolbox with guaranteed feature activation"""
    if X_columns is None:
        raise ValueError("X_columns must be provided to setup_toolbox")
        
    # Use passed feature_configs or global FEATURE_CONFIGS
    config = feature_configs if feature_configs is not None else FEATURE_CONFIGS
    
    toolbox = base.Toolbox()
    
    # Register basic attribute generator
    toolbox.register("attr_bool", np.random.randint, 0, 2)
    
    # Create base genome generator
    def create_base_genome():
        return [toolbox.attr_bool() for _ in range(TOTAL_GENE_LENGTH)]
    
    # Create individual with guaranteed features
    def create_individual():
        """Create new individual with tracking and guaranteed features"""
        genome = create_base_genome()
        
        # Activate core features
        curr_pos = 0
        for feature in X_columns:
            # Always activate important features
            if feature in CORE_FEATURES or 'AnnualIncome' in feature or 'HealthScore' in feature:
                genome[curr_pos] = 1  # Force activation
            else:
                # Random activation for other features with 30% probability
                genome[curr_pos] = 1 if random.random() < 0.3 else 0
                
            curr_pos += 1  # Move past activation bit
            
            # Handle variable length features
            if feature in FEATURE_CONFIGS and FEATURE_CONFIGS[feature]['type'] == 'variable':
                gene_length = get_gene_length(feature)
                # Initialize feature parameters randomly
                for _ in range(gene_length):
                    genome[curr_pos] = random.randint(0, 1)
                curr_pos += gene_length
        
        # Add feature combinations (pairs)
        for i in range(200):  # Number of possible pairs
            pos = curr_pos
            # Activate pair with 5% probability
            genome[pos] = 1 if random.random() < 0.05 else 0
            
            if genome[pos]:  # If pair is activated
                # Random feature indices (8 bits each)
                for _ in range(16):
                    genome[pos + 1 + _] = random.randint(0, 1)
                # Random operation (3 bits)
                for _ in range(3):
                    genome[pos + 17 + _] = random.randint(0, 1)
                # Use raw feature flag
                genome[pos + 20] = random.randint(0, 1)
                
            curr_pos += 21  # Move to next pair
        
        # Add triplet combinations
        for i in range(50):  # Number of possible triplets
            pos = curr_pos
            # Activate triplet with 2% probability
            genome[pos] = 1 if random.random() < 0.02 else 0
            
            if genome[pos]:  # If triplet is activated
                # Random feature indices (8 bits each)
                for _ in range(24):
                    genome[pos + 1 + _] = random.randint(0, 1)
                # Random operations (3 bits each)
                for _ in range(6):
                    genome[pos + 25 + _] = random.randint(0, 1)
                # Use raw feature flag
                genome[pos + 31] = random.randint(0, 1)
                
            curr_pos += 32  # Move to next triplet
        
        # Create individual with proper fitness initialization
        ind = Individual(genome=genome)
        ind.fitness = creator.FitnessMin()
        return ind
    
    # Register individual and population creation
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register genetic operators
    toolbox.register("mate", custom_crossover)
    toolbox.register("mutate", custom_mutation, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Add suspicious column check with immediate error
    suspicious_columns = [
        col for col in X_columns 
        if col == 'PremiumAmount' or  # Exact match for target
        any(term in col.lower() for term in [
            'premium', 'price', 'cost', 'payment', 'amount', 'total'
        ])
    ]
    
    if 'PremiumAmount' in suspicious_columns:
        raise ValueError("ERROR: Target variable 'PremiumAmount' found in input features!")
    elif suspicious_columns:
        print("\nWARNING: Found potentially problematic columns:")
        for col in suspicious_columns:
            print(f"- {col}")
        print("These columns might cause data leakage - consider removing them.")
    
    return toolbox
