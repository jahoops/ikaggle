import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFE, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import statsmodels.api as sm
from itertools import combinations
import argparse

class FeatureAnalyzer:
    def __init__(self, data_path: Path, output_dir: Path, validation_mode=False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = pd.read_csv(data_path)
        self.target = 'PremiumAmount'
        self.validation_mode = validation_mode
        
        if validation_mode:
            print("\nRunning in validation mode with reduced dataset...")
            self.data = self.data.sample(n=min(1000, len(self.data)), random_state=42)
            print(f"Validation data shape: {self.data.shape}")
        
        self.numerical_features = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.numerical_features = [col for col in self.numerical_features 
                                 if col not in ['id', self.target]]
        
    def generate_interaction_features(self):
        """Generate polynomial and interaction features"""
        print("\nGenerating interaction features...")
        X = self.data[self.numerical_features]
        
        # In validation mode, limit features and interactions
        if self.validation_mode:
            X = X[self.numerical_features[:5]]  # Take only first 5 features
        
        # Create and fit polynomial features
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_poly = poly.fit_transform(X)  # Fit and transform the data
        
        # Get feature names after fitting
        feature_names = poly.get_feature_names_out(
            input_features=self.numerical_features[:5] if self.validation_mode else self.numerical_features
        )
        
        # Create DataFrame with new features
        interaction_df = pd.DataFrame(X_poly, columns=feature_names)
        
        print(f"Generated {len(feature_names)} interaction features")
        
        # Calculate correlation with target for each interaction
        correlations = []
        for col in tqdm(interaction_df.columns, desc="Calculating correlations"):
            try:
                pearson, p_value = pearsonr(interaction_df[col], self.data[self.target])
                correlations.append({
                    'feature': col,
                    'correlation': abs(pearson),
                    'p_value': p_value
                })
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {col}: {str(e)}")
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        # Save results
        corr_df.to_csv(self.output_dir / 'interaction_correlations.csv', index=False)
        
        print(f"Found {len(correlations)} valid correlations")
        return interaction_df, corr_df
    
    def select_features_rfe(self, X, y, n_features=20):
        """Select important features using RFE"""
        if self.validation_mode:
            n_features = min(5, X.shape[1])  # Limit features in validation mode
            estimator = RandomForestRegressor(n_estimators=10, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        
        selector.fit(X, y)
        
        selected_features = pd.DataFrame({
            'feature': X.columns,
            'selected': selector.support_,
            'ranking': selector.ranking_
        })
        
        selected_features = selected_features.sort_values('ranking')
        selected_features.to_csv(self.output_dir / 'rfe_selected_features.csv', index=False)
        return selected_features
    
    def analyze_statistical_significance(self, X):
        """Analyze statistical significance of features"""
        print("\nAnalyzing statistical significance...")
        results = []
        
        # Reset index of target to match interaction features
        target = self.data[self.target].reset_index(drop=True)
        
        for column in tqdm(X.columns):
            try:
                # Ensure data alignment
                feature_data = X[column].reset_index(drop=True)
                
                # Remove any rows with NaN values
                valid_mask = ~(feature_data.isna() | target.isna())
                feature_clean = feature_data[valid_mask]
                target_clean = target[valid_mask]
                
                if len(feature_clean) < 2:  # Skip if not enough data
                    print(f"Warning: Not enough valid data for {column}")
                    continue
                
                # Add constant for statsmodels
                X_with_const = sm.add_constant(feature_clean)
                
                # Fit OLS model
                try:
                    model = sm.OLS(target_clean, X_with_const)
                    results_summary = model.fit()
                    
                    # Get p-value
                    p_value = results_summary.pvalues[1] if len(results_summary.pvalues) > 1 else 1.0
                    
                    results.append({
                        'feature': column,
                        'p_value': p_value,
                        'r_squared': results_summary.rsquared,
                        'valid_samples': len(feature_clean)
                    })
                except Exception as e:
                    print(f"Warning: Could not analyze {column}: {str(e)}")
                    
            except Exception as e:
                print(f"Error processing feature {column}: {str(e)}")
                continue
        
        stats_df = pd.DataFrame(results)
        if not stats_df.empty:
            stats_df = stats_df.sort_values('p_value')
            stats_df.to_csv(self.output_dir / 'statistical_significance.csv', index=False)
            print(f"\nAnalyzed {len(stats_df)} features successfully")
        else:
            print("\nWarning: No features could be analyzed")
        
        return stats_df
    
    def evaluate_model_performance(self, X, y):
        """Evaluate model performance using cross-validation"""
        if self.validation_mode:
            param_grid = {
                'n_estimators': [10],
                'max_depth': [3],
                'min_samples_split': [2]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        
        # Create base model
        base_model = RandomForestRegressor(random_state=42)
        
        # Create scorer
        scorer = make_scorer(lambda y_true, y_pred: 
                           -np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))))
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            base_model, 
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        # Save results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        cv_results.to_csv(self.output_dir / 'cross_validation_results.csv', index=False)
        
        # Save best parameters
        with open(self.output_dir / 'best_parameters.txt', 'w') as f:
            f.write(f"Best parameters: {grid_search.best_params_}\n")
            f.write(f"Best score: {grid_search.best_score_}")
        
        return grid_search.best_estimator_, grid_search.best_score_
    
    def run_analysis(self):
        """Run complete feature analysis"""
        # Generate interaction features
        interaction_df, correlations = self.generate_interaction_features()
        
        # Select features
        selected_features = self.select_features_rfe(interaction_df, self.data[self.target])
        
        # Analyze statistical significance
        stats_df = self.analyze_statistical_significance(interaction_df)
        
        # Evaluate model performance
        best_model, best_score = self.evaluate_model_performance(
            interaction_df[selected_features[selected_features['selected']]['feature']],
            self.data[self.target]
        )
        
        # Create summary report
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("Feature Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Top 10 Interaction Features by Correlation:\n")
            f.write(correlations.head(10).to_string() + "\n\n")
            
            f.write("Top 10 Features by Statistical Significance:\n")
            f.write(stats_df.head(10).to_string() + "\n\n")
            
            f.write("Selected Features by RFE:\n")
            f.write(selected_features[selected_features['selected']].to_string() + "\n\n")
            
            f.write(f"Best Model Score (RMSLE): {best_score}\n")
        
        print(f"\nAnalysis complete. Results saved to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze features for insurance prediction')
    parser.add_argument('--validate', action='store_true', 
                       help='Run in validation mode with reduced dataset and iterations')
    args = parser.parse_args()
    
    if args.validate:
        print("\nRunning quick validation of feature analysis pipeline...")
    
    try:
        analyzer = FeatureAnalyzer(
            data_path=Path('/home/USER/Projects/ikaggle/data/train.csv'),
            output_dir=Path('/home/USER/Projects/ikaggle/AAA/feature_analysis'),
            validation_mode=args.validate
        )
        
        start_time = time.time()
        analyzer.run_analysis()
        duration = time.time() - start_time
        
        print("\nAnalysis Summary:")
        print(f"- Mode: {'VALIDATION' if args.validate else 'FULL'}")
        print(f"- Duration: {duration:.2f} seconds")
        print(f"- Output directory: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
