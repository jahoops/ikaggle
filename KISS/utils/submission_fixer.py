import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

class SubmissionFixer:
    def __init__(self, results_dir='/home/USER/Projects/ikaggle/KISS/results'):
        self.results_dir = results_dir
        os.makedirs(f"{results_dir}/fixed", exist_ok=True)
        
    def load_submission(self, file_path):
        """Load a specific submission file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {file_path} with {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
    def analyze_submission(self, df, name="submission"):
        """Analyze basic statistics for a submission"""
        if df is None:
            print("No submission loaded.")
            return None
            
        pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
        
        # Compute statistics
        preds = df[pred_col].values
        stats = {
            'Name': name,
            'Min': np.min(preds),
            'Max': np.max(preds),
            'Mean': np.mean(preds),
            'Median': np.median(preds),
            'Std': np.std(preds),
            'Zero Count': np.sum(preds == 0),
            'Zero %': 100 * np.sum(preds == 0) / len(preds),
            'Very Small (<1) %': 100 * np.sum((preds > 0) & (preds < 1)) / len(preds),
            'Small (1-10) %': 100 * np.sum((preds >= 1) & (preds < 10)) / len(preds),
            'Medium (10-100) %': 100 * np.sum((preds >= 10) & (preds < 100)) / len(preds),
            'Large (100-1000) %': 100 * np.sum((preds >= 100) & (preds < 1000)) / len(preds),
            'Very Large (>1000) %': 100 * np.sum(preds >= 1000) / len(preds)
        }
        
        print("\nSubmission Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
        return stats
        
    def plot_distribution(self, df, name="submission"):
        """Plot the distribution of predictions"""
        if df is None:
            print("No submission loaded.")
            return
            
        pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
        
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        sns.histplot(df[pred_col], bins=50, kde=True)
        plt.title(f"{name} - Distribution")
        plt.xlabel("Premium Amount")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for statistics
        preds = df[pred_col].values
        plt.axvline(np.mean(preds), color='r', linestyle='--', label=f"Mean: {np.mean(preds):.2f}")
        plt.axvline(np.median(preds), color='g', linestyle='--', label=f"Median: {np.median(preds):.2f}")
        plt.legend()
        
        save_path = f"{self.results_dir}/fixed/{name}_distribution.png"
        plt.savefig(save_path)
        print(f"Saved distribution plot to {save_path}")
        plt.close()
        
    def fix_submission(self, df, method='scale', scale_factor=None, name="submission"):
        """Fix the problematic submission with various methods"""
        if df is None:
            print("No submission loaded.")
            return None
            
        pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
        fixed_df = df.copy()
        preds = df[pred_col].values
        
        # Determine the appropriate scaling factor if not provided
        if method == 'scale' and scale_factor is None:
            # Detect if the values are too small, assume typical premium is ~$500
            mean_value = np.mean(preds[preds > 0])  # Mean of non-zero values
            if mean_value < 10:  # Very small values
                scale_factor = 500 / max(mean_value, 0.01)  # Target mean of $500
                print(f"Auto-determined scale factor: {scale_factor:.2f}")
            else:
                scale_factor = 1.0
        
        # Apply the chosen method
        if method == 'scale':
            print(f"Scaling predictions by factor of {scale_factor}")
            fixed_preds = preds * scale_factor
        elif method == 'log_scale':
            print("Applying logarithmic scaling")
            # log(x+1) transformation with scaling
            non_zeros = preds[preds > 0]
            if len(non_zeros) > 0:
                min_non_zero = np.min(non_zeros)
                fixed_preds = np.log1p(preds / min_non_zero) * 100
            else:
                fixed_preds = preds
        elif method == 'rank_transform':
            print("Applying rank transformation")
            # Transform to rank percentiles, then scale to realistic premium range
            ranks = preds.argsort().argsort() / len(preds)  # 0 to 1
            fixed_preds = 100 + 900 * ranks  # Scale to $100-$1000 range
        else:
            raise ValueError(f"Unknown fix method: {method}")
            
        # Additional fixes
        # 1. Ensure all premiums are non-negative
        fixed_preds = np.maximum(fixed_preds, 0)
        
        # 2. For zero values, apply a minimum premium if needed
        if np.sum(fixed_preds == 0) > 0:
            print(f"Found {np.sum(fixed_preds == 0)} zeros - keeping as is.")
        
        # 3. Cap unreasonably high values
        high_threshold = 10000  # $10,000
        if np.any(fixed_preds > high_threshold):
            print(f"Capping {np.sum(fixed_preds > high_threshold)} values above ${high_threshold}")
            fixed_preds = np.minimum(fixed_preds, high_threshold)
            
        # Update the DataFrame
        fixed_df[pred_col] = fixed_preds
        
        # Save the fixed submission
        fixed_file = f"{self.results_dir}/fixed/fixed_{name}"
        fixed_df.to_csv(fixed_file, index=False)
        print(f"Saved fixed submission to {fixed_file}")
        
        # Show before/after stats
        print("\nOriginal submission stats:")
        self.analyze_submission(df, name=f"original_{name}")
        
        print("\nFixed submission stats:")
        self.analyze_submission(fixed_df, name=f"fixed_{name}")
        
        return fixed_df
        
    def fix_group_ensemble(self, file_path):
        """Specifically fix the group ensemble file with appropriate scaling"""
        df = self.load_submission(file_path)
        if df is None:
            return None
            
        name = os.path.basename(file_path)
        
        # Analyze before fixing
        self.analyze_submission(df, name=name)
        self.plot_distribution(df, name=f"before_{name}")
        
        # Fix with auto-scaling
        fixed_df = self.fix_submission(df, method='scale', name=name)
        
        # Plot the fixed distribution
        self.plot_distribution(fixed_df, name=f"after_{name}")
        
        return fixed_df


if __name__ == "__main__":
    fixer = SubmissionFixer()
    # Fix the problematic ensemble file
    problem_file = '/home/USER/Projects/ikaggle/KISS/results/final_group_ensemble_submission.csv'
    fixed_df = fixer.fix_group_ensemble(problem_file)
    
    # You can also try different fixing methods:
    # fixer.fix_submission(fixed_df, method='rank_transform', name="rank_transformed")
