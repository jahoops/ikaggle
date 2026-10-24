import numpy as np
from ikaggle.AAA.experiments.feature_engineering import decode_individual
from ikaggle.AAA.utils.data_loading import GENOME_TEMPLATE, TOTAL_GENE_LENGTH
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS

def print_genome(individual, X_full):
    """Print detailed genome structure.
       Minimal stub: Print the individual mapping.
    """
    mapping = decode_individual(individual, X_full)
    for feat, value in mapping.items():
        print(f"{feat}: {value}")


    total_features = len(mapping)
    print(f"Total features: {total_features}")
    print(f"Total genome length: {TOTAL_GENE_LENGTH}")
    print(f"Active features: {total_features}/{TOTAL_GENE_LENGTH}")
    print(f"Active features: {total_features}/{TOTAL_GENE_LENGTH}")
    print("Genome structure:")  # Print genome structure
    for i, gene in enumerate(individual):
        print(f"{GENOME_TEMPLATE[i][0]}: {gene}")