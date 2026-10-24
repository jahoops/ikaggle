import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

class PredictionSelector:
    def __init__(self, feature_importance_threshold: float = 0.01):
        self.threshold = feature_importance_threshold
        self.meta_model = None
        self.feature_importance: Dict[str, float] = {}
        self.best_features: List[str] = []
        
    def train_selector(self, 
                      all_predictions: pd.DataFrame, 
                      actual_values: pd.Series, 
                      feature_values: pd.DataFrame) -> None:
        """Train a model to select the best prediction for each case"""
        # Create meta-features
        meta_features = self._create_meta_features(all_predictions, feature_values)
            
        # Train meta-model
        self.meta_model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            random_state=42
        )
        
        # Fit and get feature importance
        self.meta_model.fit(meta_features, actual_values)
        self.feature_importance = dict(zip(
            meta_features.columns,
            self.meta_model.feature_importances_
        ))
        
        # Filter important features
        self.best_features = [
            feat for feat, imp in self.feature_importance.items()
            if imp > self.threshold
        ]
        
    def predict(self, all_predictions: pd.DataFrame, feature_values: pd.DataFrame) -> pd.Series:
        """Generate final predictions using meta-model"""
        meta_features = self._create_meta_features(all_predictions, feature_values)
        return pd.Series(
            self.meta_model.predict(meta_features[self.best_features]),
            index=feature_values.index
        )
        
    def save_submission(self, predictions: pd.Series, sample_ids: pd.Series, experiment_name: str) -> Path:
        """Save final predictions to CSV
        
        Args:
            predictions: Series with predictions
            sample_ids: Series with actual sample IDs
            experiment_name: Name of experiment for filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"segment_output/{experiment_name}_submission_{timestamp}.csv")
        output_path.parent.mkdir(exist_ok=True)
        
        # Create submission dataframe with proper column names and sample IDs
        submission_df = pd.DataFrame({
            "id": sample_ids.values,  # Use actual sample IDs
            "PremiumAmount": predictions.values 
        })
        
        # Save with proper header format
        submission_df.to_csv(output_path, index=False)
        
        print(f"\nSaved submission to: {output_path}")
        
        return output_path
        
    def validate_selector(self, 
                         all_predictions: pd.DataFrame,
                         actual_values: pd.Series,
                         feature_values: pd.DataFrame) -> Dict[str, float]:
        """Validate meta-model performance
        
        Args:
            all_predictions: DataFrame with predictions from each model
            actual_values: True target values
            feature_values: Original features used for meta-features
            
        Returns:
            Dictionary with validation metrics
        """
        if self.meta_model is None:
            raise RuntimeError("Meta-model must be trained before validation")
            
        # Generate predictions
        meta_predictions = self.predict(all_predictions, feature_values)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_log_error, mean_squared_error
        metrics = {
            'meta_rmsle': float(np.sqrt(mean_squared_log_error(actual_values, meta_predictions))),
            'meta_rmse': float(np.sqrt(mean_squared_error(actual_values, meta_predictions))),
            'val_samples': len(actual_values)
        }
        
        # Compare with individual model performance
        for model_name in all_predictions.columns:
            metrics[f'{model_name}_rmsle'] = float(np.sqrt(
                mean_squared_log_error(actual_values, all_predictions[model_name])
            ))
        
        print("\nMeta-model validation metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics

    def _create_meta_features(self, predictions: pd.DataFrame, feature_values: pd.DataFrame) -> pd.DataFrame:
        """Create meta-features from base model predictions and original features
        
        Args:
            predictions: DataFrame containing predictions from each model
            feature_values: Original feature values
            
        Returns:
            DataFrame containing meta-features
        """
        meta_features = pd.DataFrame()
        
        # Add original predictions as features
        meta_features = predictions.copy()
        
        # Add statistics of predictions
        meta_features['pred_mean'] = predictions.mean(axis=1)
        meta_features['pred_std'] = predictions.std(axis=1)
        meta_features['pred_min'] = predictions.min(axis=1)
        meta_features['pred_max'] = predictions.max(axis=1)
        
        # Add original features
        for col in feature_values.columns:
            if col != 'id' and col != 'PremiumAmount':
                meta_features[f'orig_{col}'] = feature_values[col]
        
        return meta_features