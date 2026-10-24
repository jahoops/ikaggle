import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import seaborn as sns

class DiversityChecker:
    """
    Utility class to check and improve diversity among model predictions
    """
    def __init__(self):
        self.models = []
        self.predictions = {}
        self.correlations = None
        
    def add_submission(self, path, name=None):
        """Add a submission file to the diversity checker"""
        try:
            df = pd.read_csv(path)
            if name is None:
                name = os.path.basename(path)
            
            self.models.append(name)
            self.predictions[name] = df['Premium Amount'].values
            print(f"Added {name} with {len(df)} predictions")
            return True
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return False
    
    def calculate_correlations(self):
        """Calculate correlation matrix between all model predictions"""
        if not self.models:
            print("No models added yet")
            return
            
        corr_matrix = np.zeros((len(self.models), len(self.models)))
        
        for i, model_i in enumerate(self.models):
            for j, model_j in enumerate(self.models):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    corr, _ = pearsonr(self.predictions[model_i], self.predictions[model_j])
                    corr_matrix[i, j] = corr
                    
        self.correlations = pd.DataFrame(
            corr_matrix, 
            index=self.models,
            columns=self.models
        )
        return self.correlations
    
    def visualize_correlations(self, save_path=None):
        """Visualize correlation matrix between models"""
        if self.correlations is None:
            self.calculate_correlations()
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.correlations, annot=True, cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Model Prediction Correlations')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Correlation matrix saved to {save_path}")
        else:
            plt.show()
    
    def calculate_diversity_score(self):
        """Calculate a diversity score (lower correlations = higher diversity)"""
        if self.correlations is None:
            self.calculate_correlations()
            
        # Calculate mean correlation excluding self-correlations
        n = len(self.models)
        total_corr = self.correlations.sum().sum() - n  # Subtract diagonal (self-correlations)
        mean_corr = total_corr / (n * n - n)  # n² - n is number of non-diagonal elements
        
        # Diversity score is inverse of correlation (1 = perfect diversity, 0 = no diversity)
        diversity = 1 - mean_corr
        
        print(f"Ensemble diversity score: {diversity:.4f} (higher is better)")
        return diversity
    
    def visualize_distributions(self, bins=50, save_path=None):
        """Visualize the distribution of predictions from each model"""
        if not self.models:
            print("No models added yet")
            return
            
        plt.figure(figsize=(14, 8))
        
        for model in self.models:
            plt.hist(
                self.predictions[model], 
                alpha=0.4, 
                bins=bins, 
                label=model,
                density=True
            )
            
        plt.legend()
        plt.title('Distribution of Model Predictions')
        plt.xlabel('Predicted Premium Amount')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        else:
            plt.show()
    
    def optimize_weights(self, target_diversity=0.3):
        """Optimize ensemble weights to increase diversity"""
        if self.correlations is None:
            self.calculate_correlations()
        
        # Start with equal weights
        n = len(self.models)
        weights = np.ones(n) / n
        
        # Simple gradient descent to maximize diversity
        learning_rate = 0.01
        iterations = 100
        best_weights = weights.copy()
        best_diversity = 0
        
        for _ in range(iterations):
            for i in range(n):
                # Calculate gradient direction (very simplified)
                weights_up = weights.copy()
                weights_up[i] *= 1.05
                weights_up = weights_up / np.sum(weights_up)  # Normalize
                
                # Calculate diversity with new weights
                weighted_corr = 0
                for j in range(n):
                    for k in range(n):
                        if j != k:
                            weighted_corr += weights_up[j] * weights_up[k] * self.correlations.iloc[j, k]
                            
                diversity_up = 1 - weighted_corr
                
                # If diversity improved, keep the new weights
                if diversity_up > best_diversity:
                    best_diversity = diversity_up
                    best_weights = weights_up.copy()
                    
            weights = best_weights.copy()
            
        # Normalize and return
        best_weights = best_weights / np.sum(best_weights)
        
        result = {
            'model': self.models,
            'weight': best_weights,
            'diversity_score': best_diversity
        }
        
        print(f"Optimized weights for diversity (score: {best_diversity:.4f}):")
        for model, weight in zip(self.models, best_weights):
            print(f"  {model}: {weight:.4f}")
            
        return result
    
    def get_optimized_ensemble(self):
        """Create optimized ensemble prediction"""
        if not self.models:
            print("No models added yet")
            return None
            
        weights = self.optimize_weights()['weight']
        ensemble = np.zeros_like(next(iter(self.predictions.values())))
        
        for i, model in enumerate(self.models):
            ensemble += weights[i] * self.predictions[model]
            
        return ensemble

def optimize_ensemble_diversity(submission_files, output_path=None):
    """
    Optimize ensemble weights for maximum prediction diversity
    
    Args:
        submission_files: List of submission file paths
        output_path: Path to save the optimized ensemble
        
    Returns:
        Dictionary with weights and diversity score
    """
    checker = DiversityChecker()
    
    # Add all submissions
    for file in submission_files:
        checker.add_submission(file)
    
    # Analyze diversity
    checker.calculate_correlations()
    initial_diversity = checker.calculate_diversity_score()
    
    # Visualize
    viz_dir = os.path.dirname(output_path) if output_path else '/home/USER/Projects/ikaggle/KISS/results'
    os.makedirs(viz_dir, exist_ok=True)
    
    checker.visualize_correlations(save_path=f"{viz_dir}/model_correlations.png")
    checker.visualize_distributions(save_path=f"{viz_dir}/prediction_distributions.png")
    
    # Optimize weights
    weights = checker.optimize_weights()
    optimized_ensemble = checker.get_optimized_ensemble()
    
    # Save optimized ensemble
    if output_path:
        # Get IDs from the first submission file
        ids = pd.read_csv(submission_files[0])['id'].values
        
        pd.DataFrame({
            'id': ids,
            'Premium Amount': optimized_ensemble
        }).to_csv(output_path, index=False)
        
        print(f"Optimized ensemble saved to {output_path}")
    
    return {
        'weights': dict(zip(checker.models, weights['weight'])),
        'initial_diversity': initial_diversity,
        'optimized_diversity': weights['diversity_score']
    }

if __name__ == "__main__":
    import glob
    
    # Test with all submission files in the results directory
    submission_files = glob.glob("/home/USER/Projects/ikaggle/KISS/results/submission_model_*.csv")
    output_path = "/home/USER/Projects/ikaggle/KISS/results/diversity_optimized_ensemble.csv"
    
    result = optimize_ensemble_diversity(submission_files, output_path)
    print(f"Diversity improvement: {result['initial_diversity']:.4f} → {result['optimized_diversity']:.4f}")
