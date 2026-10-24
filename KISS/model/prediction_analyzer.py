import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

class PredictionAnalyzer:
    """
    Utility to analyze and diagnose prediction issues
    """
    def __init__(self, results_dir='/home/USER/Projects/ikaggle/KISS/results'):
        self.results_dir = results_dir
        os.makedirs(f"{results_dir}/diagnostics", exist_ok=True)
        self.submissions = {}
    
    def load_submissions(self, pattern=None):
        """Load submission files matching a pattern"""
        if pattern is None:
            pattern = f"{self.results_dir}/*submission*.csv"
            
        files = glob.glob(pattern)
        print(f"Found {len(files)} submission files")
        
        for file in files:
            name = os.path.basename(file)
            try:
                self.submissions[name] = pd.read_csv(file)
                print(f"Loaded {name} with {len(self.submissions[name])} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        return len(self.submissions)
    
    def analyze_submissions(self):
        """Analyze statistics for each submission"""
        if not self.submissions:
            print("No submissions loaded.")
            return
        
        stats = []
        
        for name, df in self.submissions.items():
            pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
            
            # Basic statistics
            preds = df[pred_col].values
            stats.append({
                'Name': name,
                'Min': np.min(preds),
                'Max': np.max(preds),
                'Mean': np.mean(preds),
                'Median': np.median(preds),
                'Std': np.std(preds),
                'Negative %': 100 * np.sum(preds < 0) / len(preds),
                'Zero %': 100 * np.sum(preds == 0) / len(preds),
                'Very Small %': 100 * np.sum((preds > 0) & (preds < 1)) / len(preds),
                'Very Large %': 100 * np.sum(preds > 5000) / len(preds)
            })
        
        # Print and save as table
        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(f"{self.results_dir}/diagnostics/submission_stats.csv", index=False)
        
        print("\nSubmission Statistics:")
        print(stats_df.to_string(index=False, float_format='{:.2f}'.format))
        
        return stats_df
    
    def visualize_distributions(self):
        """Visualize prediction distributions"""
        if not self.submissions:
            print("No submissions loaded.")
            return
            
        # Create subplot grid based on number of submissions
        n = len(self.submissions)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if n == 1:
            axes = np.array([axes])  # Convert to array for consistent indexing
        axes = axes.flatten()
        
        # Plot histograms for each submission
        for i, (name, df) in enumerate(self.submissions.items()):
            if i < len(axes):
                pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
                sns.histplot(df[pred_col], ax=axes[i], bins=50, log_scale=(False, True))
                axes[i].set_title(f"{name} - Distribution")
                axes[i].set_xlabel("Premium")
                
                # Add vertical lines for common statistics
                preds = df[pred_col].values
                axes[i].axvline(np.mean(preds), color='r', linestyle='--', label=f"Mean: {np.mean(preds):.2f}")
                axes[i].axvline(np.median(preds), color='g', linestyle='--', label=f"Median: {np.median(preds):.2f}")
                axes[i].legend()
        
        # Hide any unused subplots
        for i in range(n, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/diagnostics/prediction_distributions.png")
        print(f"Saved prediction distributions to {self.results_dir}/diagnostics/prediction_distributions.png")
        
        # Create a boxplot of all submissions for comparison
        plt.figure(figsize=(10, 6))
        all_dfs = []
        for name, df in self.submissions.items():
            pred_col = 'Premium Amount' if 'Premium Amount' in df.columns else 'PremiumAmount'
            temp_df = pd.DataFrame({
                'Model': name,
                'Prediction': df[pred_col].values
            })
            all_dfs.append(temp_df)
            
        combined = pd.concat(all_dfs)
        # Clip values for better visualization
        combined['Prediction'] = np.clip(combined['Prediction'], -10, 1000)
        
        sns.boxplot(x='Model', y='Prediction', data=combined)
        plt.xticks(rotation=90)
        plt.title('Premium Prediction Comparison')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/diagnostics/prediction_boxplots.png")
        print(f"Saved prediction boxplots to {self.results_dir}/diagnostics/prediction_boxplots.png")
        
    def run_diagnostics(self):
        """Run all diagnostics"""
        self.load_submissions()
        self.analyze_submissions()
        self.visualize_distributions()
        print("All diagnostics completed.")
        

if __name__ == "__main__":
    analyzer = PredictionAnalyzer()
    analyzer.run_diagnostics()
