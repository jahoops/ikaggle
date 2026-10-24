import unittest
import numpy as np
import pandas as pd
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS, get_gene_length, decode_gene
from ikaggle.AAA.utils.data_loading import GENOME_TEMPLATE, TOTAL_GENE_LENGTH
from ikaggle.AAA.experiments.genome_ops import (
    decode_individual,
    decode_gene
)

class TestGenomeOps(unittest.TestCase):
    def setUp(self):
        # Create minimal test data
        self.X_full = pd.DataFrame({
            'Age': [25, 35, 45],
            'Income': [1000, 2000, 3000],
            'Score': [0.5, 0.7, 0.9]
        })
        self.X_full_scaled = (self.X_full - self.X_full.mean()) / self.X_full.std()

    def test_genome_template_creation(self):
        template = GENOME_TEMPLATE
        self.assertIsNotNone(template)
        self.assertTrue(len(template) > 0)
        
        # Check base features are included
        base_features = [t['name'] for t in template if t['type'] == 'activation']
        for col in self.X_full.columns:
            self.assertIn(col, base_features)

    def test_individual_decoding(self):
        # Create a simple test individual
        template = GENOME_TEMPLATE
        individual = [1] * len(template)  # All features active
        
        decoded = decode_individual(individual, self.X_full)
        self.assertIsNotNone(decoded)
        
        # Check all base features are present
        for col in self.X_full.columns:
            self.assertIn(col, decoded)

    def test_gene_decoding(self):
        # Test decoding of a specific feature gene
        feature_name = 'Age'
        gene = [1, 1, 0, 1]  # Example gene sequence
        value = decode_gene(gene, feature_name, 0)
        
        self.assertIsNotNone(value)
        self.assertIsInstance(value, (int, float))
        self.assertTrue(0 <= value <= 100)  # Assuming age range

if __name__ == '__main__':
    unittest.main()
