from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import cudf
import cupy as cp
from cuml.ensemble import RandomForestRegressor
from datetime import datetime
from .base_segment_experiment import BaseSegmentExperiment

class RapidsSegmentExperiment(BaseSegmentExperiment):
    """RAPIDS-based segment experiment with GPU support"""
    
    def __init__(self, output_dir: Path, test_mode: bool = False):
        self.numeric_features = [
            'Age',
            'AnnualIncome', 
            'HealthScore',
            'CreditScore',
            'NumberofDependents'
        ]
        self.feature_columns = self.numeric_features.copy()
        self.feature_order = self.feature_columns.copy()
        self.model = None
        
        # Initialize base class
        super().__init__(output_dir, test_mode)

    def get_quick_test_params(self) -> Dict[str, Any]:
        """Parameters for quick testing"""
        return {
            'n_estimators': 100,
            'max_depth': 8,
            'n_bins': 128,
            'n_streams': 4,
            'max_features': 'auto',
            'max_samples': 0.7
        }
    
    def get_model_params(self) -> Dict[str, Any]:
        """Parameters optimized for insurance premium prediction"""
        return {
            'n_estimators': 1000,          # More trees for better accuracy
            'max_depth': 15,               # Deeper trees for complex relationships
            'n_bins': 256,                 # More bins for better splits
            'n_streams': 8,                # Parallel processing
            'max_features': 'sqrt',        # Standard RF feature sampling
            'max_samples': 0.8,            # Bagging fraction
            'min_samples_leaf': 2,         # Prevent overfitting
            'min_samples_split': 6,        # Minimum samples for split
            'bootstrap': True,             # Enable bootstrapping
            'random_state': 42,            # Reproducibility
            'max_leaves': -1,              # Allow full tree growth
            'accuracy_metric': 'rmse'      # Relevant for regression
        }

    def create_model(self) -> RandomForestRegressor:
        """Create RAPIDS Random Forest model with appropriate parameters"""
        params = self.get_quick_test_params() if self.test_mode else self.get_model_params()
        return RandomForestRegressor(**params)

    def train(self, train_data: pd.DataFrame) -> None:
        """Train RAPIDS model with GPU acceleration"""
        # Convert to GPU DataFrame with float32 dtype
        X = cudf.DataFrame(train_data[self.feature_columns], dtype='float32')
        y = cudf.Series(train_data['PremiumAmount'], dtype='float32')
        
        # Create and train model
        self.model = self.create_model()
        self.model.fit(X, y)

    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model and return metrics"""
        if self.model is None:
            raise RuntimeError("Model must be trained before validation")
        
        # Convert to GPU with float32 dtype
        X_val = cudf.DataFrame(val_data[self.feature_columns], dtype='float32')
        y_val = val_data['PremiumAmount'].values
        ids = val_data['id'].values
        
        # Generate predictions
        y_pred = self.model.predict(X_val)
        
        # Convert predictions to CPU numpy array
        if isinstance(y_pred, (cudf.Series, cudf.DataFrame)):
            y_pred = y_pred.to_numpy()
        elif hasattr(y_pred, 'get'):
            y_pred = y_pred.get()
        
        # Save predictions
        save_path = self._save_predictions(ids, y_pred, 'validation')
        print(f"Saved validation predictions to: {save_path}")
        
        return self.calculate_metrics(y_val, y_pred, 'validation')

    def test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Generate test predictions"""
        if self.model is None:
            raise RuntimeError("Model must be trained before testing")
        
        # Convert to GPU with float32 dtype
        X_test = cudf.DataFrame(test_data[self.feature_columns], dtype='float32')
        ids = test_data['id'].values
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Convert predictions to CPU numpy array
        if isinstance(y_pred, (cudf.Series, cudf.DataFrame)):
            y_pred = y_pred.to_numpy()
        elif hasattr(y_pred, 'get'):
            y_pred = y_pred.get()
        
        # Save predictions
        save_path = self._save_predictions(ids, y_pred, 'test')
        print(f"Saved test predictions to: {save_path}")
        
        return {'test_samples': len(y_pred)}

    def _save_predictions(self, ids: np.ndarray, preds: np.ndarray, dataset_type: str) -> Path:
        """Save predictions to standardized location"""
        preds_dir = self.output_dir / 'predictions'
        preds_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = preds_dir / f'{dataset_type}_predictions.csv'
        pd.DataFrame({
            'id': ids,
            'PremiumAmount': preds
        }).to_csv(save_path, index=False)
        
        return save_path

    def get_experiment_metadata(self) -> Dict:
        """Return metadata including GPU metrics and model information"""
        metadata = {
            "name": "RapidsSegmentExperiment",
            "test_mode": self.test_mode,
            "timestamp": datetime.now().isoformat(),
            "hardware_config": {
                "gpu_enabled": True,
                "device_name": None,
                "memory_allocated": None,
                "compute_capability": None
            }
        }
        
        # Add GPU metrics
        try:
            import cupy as cp
            dev = cp.cuda.Device()
            props = cp.cuda.runtime.getDeviceProperties(dev.id)
            free_memory, total_memory = dev.mem_info
            
            metadata["hardware_config"].update({
                "device_name": props['name'].decode('utf-8'),
                "memory_allocated": float((total_memory - free_memory) / (1024**2)),
                "memory_available": float(free_memory / (1024**2)),
                "total_memory": float(total_memory / (1024**2)),
                "compute_capability": f"{props['major']}.{props['minor']}"
            })
        except Exception as e:
            print(f"Warning: Could not get GPU metrics: {e}")
        
        # Add model information if available
        if self.model is not None:
            metadata["model_info"] = {
                "n_estimators": self.model.n_estimators,
                "max_depth": self.model.max_depth,
                "n_features": len(self.feature_columns)
            }
        
        return metadata