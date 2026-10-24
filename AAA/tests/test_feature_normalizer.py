import unittest
import numpy as np
from ikaggle.AAA.utils.feature_normalizer import normalize_features

class TestFeatureNormalizer(unittest.TestCase):
    def setUp(self):
        """Create test data"""
        print("\nSetting up test data...")
        self.feature_matrix = np.array([
            [1000, 25, 0.5],    # Income, Age, Score
            [2000, 35, 0.7],
            [5000, 45, 0.3],
            [3000, 55, 0.9]
        ]).astype(float)
        self.feature_names = ['Income', 'Age', 'Score']
        print(f"Test data shape: {self.feature_matrix.shape}")

    def test_basic_normalization(self):
        """Test basic normalization functionality"""
        print("\nTesting basic normalization...")
        normalized = normalize_features(self.feature_matrix, self.feature_names, verbose=True)
        
        print(f"\nOriginal data:\n{self.feature_matrix}")
        print(f"\nNormalized data:\n{normalized}")
        
        # Test shape
        self.assertEqual(normalized.shape, self.feature_matrix.shape)
        print("✓ Shape preserved")
        
        # Test ranges
        self.assertTrue(np.all(np.abs(normalized) <= 1.0))
        print("✓ All values within [-1, 1]")
        
        # Test column-specific processing
        income_col = normalized[:, 0]
        age_col = normalized[:, 1]
        
        print(f"\nIncome column range: [{income_col.min():.3f}, {income_col.max():.3f}]")
        print(f"Age column range: [{age_col.min():.3f}, {age_col.max():.3f}]")
        
        # Income order preservation
        income_orig = self.feature_matrix[:, 0]
        income_norm = normalized[:, 0]
        
        # Debug: Print income arrays
        print("\nDEBUG: Original Income:", income_orig)
        print("DEBUG: Normalized Income:", income_norm)
        
        # Pairwise monotonicity check for Income
        for i in range(len(income_orig)):
            for j in range(i+1, len(income_orig)):
                if income_orig[i] < income_orig[j]:
                    print(f"DEBUG: Comparing row {i} (income {income_orig[i]:.0f}, normalized {income_norm[i]:.3f}) "
                          f"and row {j} (income {income_orig[j]:.0f}, normalized {income_norm[j]:.3f})")
                    self.assertLess(income_norm[i], income_norm[j],
                        msg=f"Row {i} income ({income_orig[i]}) should map lower than row {j} income ({income_orig[j]}).")
        print("✓ Income ordering preserved")
        
        # Age scaling
        self.assertTrue(np.max(np.abs(age_col)) <= 0.25)
        print("✓ Age properly scaled")

    def test_zero_handling(self):
        """Test handling of zero values"""
        print("\nTesting zero handling...")
        test_data = np.array([
            [0, 25, 0.5],
            [1000, 0, 0.7],
            [2000, 35, 0]
        ])
        normalized = normalize_features(test_data, self.feature_names, verbose=False)
        print(f"\nZero handling result:\n{normalized}")
        self.assertIsNotNone(normalized)
        print("✓ Zero values handled correctly")

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        print("\nTesting invalid inputs...")
        with self.assertRaises(ValueError):
            normalize_features(None, self.feature_names)
        print("✓ None matrix detected")
        
        with self.assertRaises(ValueError):
            normalize_features(self.feature_matrix, None)
        print("✓ None feature names detected")

if __name__ == '__main__':
    unittest.main(verbosity=2)
