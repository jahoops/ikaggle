from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import cudf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from .base_segment_experiment import BaseSegmentExperiment

class SVRSegmentExperiment(BaseSegmentExperiment):
    """SVR-based segment experiment with GPU support"""
    
    def __init__(self, output_dir: Path, test_mode: bool = False):
        # Define features before super().__init__
        self.numeric_features = [
            'Age',
            'AnnualIncome', 
            'HealthScore',
            'CreditScore',
            'NumberofDependents'
        ]
        self.feature_columns = self.numeric_features.copy()
        self.feature_order = self.feature_columns.copy()  # Add feature order
        self.model = None
        self.model_trainer = None
        
        # Initialize base class
        super().__init__(output_dir, test_mode)
        
        # Initialize GPU support
        self.use_gpu = self._check_gpu_available()

    def get_quick_test_params(self) -> Dict[str, Any]:
        """Return parameters optimized for quick testing"""
        base_params = {
            'C': 1.0,  # Reduced from 10.0 to help convergence
            'epsilon': 0.2,  # Increased epsilon for faster convergence
            'max_iter': 10000,  # Further increased iterations
            'tol': 1e-2,  # More relaxed tolerance
            'kernel': 'rbf',
            'verbose': True,
            'cache_size': 4000,  # Increased cache size
            'shrinking': True,  # Enable shrinking heuristic
        }
        
        if self.use_gpu:
            gpu_params = base_params.copy()
            gpu_params.update({
                'gamma': 'auto'
            })
            return gpu_params
        else:
            cpu_params = base_params.copy()
            cpu_params.update({
                'gamma': 'auto',
                'random_state': 42
            })
            return cpu_params

    def create_model(self) -> BaseEstimator:
        """Create GPU-based SVR model"""
        params = self.get_quick_test_params() if self.test_mode else self.get_model_params()
        
        try:
            from cuml.svm import SVR as cuSVR
            return cuSVR(**params)
        except ImportError as e:
            raise RuntimeError("GPU-based SVR implementation required") from e
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using GPU acceleration"""
        # First apply base preprocessing
        data = super().preprocess_data(data)
        result = data.copy()
        
        # Get feature columns
        feature_cols = [col for col in data.columns 
                       if col not in ['id', 'PremiumAmount']]
        
        # Initialize GPU scalers if not exists
        if not hasattr(self, 'X_scaler'):
            from cuml.preprocessing import StandardScaler as cuStandardScaler
            self.X_scaler = cuStandardScaler()
                
        if not hasattr(self, 'y_scaler'):
            from cuml.preprocessing import StandardScaler as cuStandardScaler
            self.y_scaler = cuStandardScaler()
        
        # Convert to GPU DataFrame using cuDF
        import cudf
        X = cudf.DataFrame(data[feature_cols])
        
        # Scale features using cuML scaler
        X_scaled = self.X_scaler.fit_transform(X)
        X_scaled = cudf.DataFrame(X_scaled).clip(-3, 3)  # Clip outliers
        
        # Scale target if present
        if 'PremiumAmount' in data.columns:
            y = cudf.Series(data['PremiumAmount'])
            y_arr = cudf.Series(np.log1p(y.values)).values.reshape(-1, 1)
            y_scaled = self.y_scaler.fit_transform(y_arr)
            result['PremiumAmount'] = y_scaled
        
        # Update DataFrame with scaled values
        for col, values in zip(feature_cols, X_scaled.values.T):
            result[col] = values
        
        return result
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions ensuring non-negative values"""
        if self.feature_order is None:
            raise ValueError("Model must be trained before prediction")
            
        # Verify all required features are present
        missing_cols = set(self.feature_order) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")
        
        # Preserve ID column if present
        id_values = data['id'].copy() if 'id' in data.columns else None
        
        # Preprocess features with same column order as training
        X = self.preprocess_data(data[self.feature_order])
        
        # Make predictions and ensure non-negative values
        predictions = self.model.predict(X)
        
        # Inverse transform predictions (including log transform)
        if hasattr(self, 'y_scaler'):
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1))
            predictions = np.expm1(predictions)  # Inverse of log1p
        
        predictions = np.clip(predictions, 1e-6, None)
        
        # Create submission format
        result = pd.DataFrame({'PremiumAmount': predictions})
        if id_values is not None:
            result['id'] = id_values
        
        return result[['id', 'PremiumAmount']]

    def train(self, train_data: pd.DataFrame) -> None:
        """Train SVR model with hyperparameter optimization"""
        # Ensure we only use numeric features
        X = train_data[self.feature_columns].copy()
        
        if 'id' in X.columns:
            X = X.drop('id', axis=1)
            
        if 'PremiumAmount' not in train_data.columns:
            raise ValueError("Target column 'PremiumAmount' not found in training data")
            
        y = train_data['PremiumAmount'].values
        
        # Scale features and target
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y.reshape(-1, 1))
        
        # Optimize hyperparameters
        self.model = self.optimize_hyperparameters(
            pd.DataFrame(X_scaled, columns=self.feature_columns), 
            pd.Series(y_scaled.ravel())
        )

    def get_experiment_metadata(self) -> Dict:
        """Return metadata for the experiment including GPU metrics"""
        metadata = {
            "name": "SVRSegmentExperiment",
            "test_mode": self.test_mode,
            "timestamp": datetime.now().isoformat(),
            "hardware_config": {
                "gpu_enabled": self.use_gpu,
                "device_name": None,
                "memory_allocated": None,
                "compute_capability": None
            }
        }
        
        # Add GPU metrics if available
        if self.use_gpu:
            try:
                import cupy as cp
                dev = cp.cuda.Device()
                
                # Get device properties using runtime API
                try:
                    props = cp.cuda.runtime.getDeviceProperties(dev.id)
                    
                    # Get memory info - use mem_info as a property
                    free_memory, total_memory = dev.mem_info  # Access as property
                    memory_used = total_memory - free_memory
                    
                    metadata["hardware_config"].update({
                        "device_name": props['name'].decode('utf-8'),
                        "memory_allocated": float(memory_used / (1024**2)),  # MB
                        "memory_available": float(free_memory / (1024**2)),  # MB
                        "total_memory": float(total_memory / (1024**2)),  # MB
                        "compute_capability": f"{props['major']}.{props['minor']}"
                    })
                except Exception as e:
                    print(f"Warning: Could not get detailed GPU properties: {e}")
                    # Fallback with basic information
                    free_memory, total_memory = dev.mem_info  # Access as property
                    metadata["hardware_config"].update({
                        "device_name": f"CUDA Device {dev.id}",
                        "memory_allocated": float((total_memory - free_memory) / (1024**2)),
                        "memory_available": float(free_memory / (1024**2)),
                        "total_memory": float(total_memory / (1024**2))
                    })
            except ImportError:
                pass

        # Add model coefficients and performance metrics
        if hasattr(self, 'model'):
            metadata.update({
                "model_params": {
                    "C": float(self.model.C),
                    "epsilon": float(self.model.epsilon),
                    "n_features": len(self.feature_order) if self.feature_order else 0
                },
                "feature_coefficients": self._get_feature_coefficients()
            })
        
        return metadata

    def _get_feature_coefficients(self) -> Dict[str, float]:
        """Get feature coefficients from trained SVR model"""
        if hasattr(self, 'model') and hasattr(self.model, 'coef_'):
            return {
                feat: float(coef) 
                for feat, coef in zip(self.feature_order, self.model.coef_)
            }
        return {}

    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available for SVR"""
        try:
            import cudf
            import cupy
            from cuml.svm import SVR as cuSVR
            
            # Try to create a small test array on GPU
            test_array = cupy.array([1, 2, 3])
            del test_array
            
            print("GPU support confirmed")
            return True
            
        except ImportError as e:
            raise RuntimeError("GPU support required but not available") from e

    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model and use base class metrics calculation"""
        if self.model is None:
            raise RuntimeError("Model must be trained before validation")
            
        # Prepare validation data
        X_val = val_data[self.feature_columns].copy()
        y_val = val_data['PremiumAmount'].values
        
        # Generate predictions
        y_pred = self.model.predict(X_val)
        
        # Use base class metrics calculation
        return self.calculate_metrics(y_val, y_pred, 'validation')

    def test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Test model and use base class metrics calculation"""
        if self.model is None:
            raise RuntimeError("Model must be trained before testing")
            
        # Prepare test data
        X_test = test_data[self.feature_columns].copy()
        y_test = test_data['PremiumAmount'].values
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Use base class metrics calculation
        return self.calculate_metrics(y_test, y_pred, 'test')

    def get_param_distributions(self) -> Dict[str, List[Any]]:
        """Define full hyperparameter search space for SVR"""
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'tol': [1e-4, 1e-3, 1e-2],
            'cache_size': [2000, 4000, 8000],
            'max_iter': [10000, 20000]
        }

    def get_quick_param_distributions(self) -> Dict[str, List[Any]]:
        """Define minimal parameter search space for testing"""
        return {
            'C': [1.0],
            'epsilon': [0.1],
            'gamma': ['scale'],
            'tol': [1e-3],
            'cache_size': [2000],
            'max_iter': [5000]
        }

    def get_experiment_metadata(self) -> Dict:
        """Return metadata including hyperparameter optimization results"""
        metadata = super().get_experiment_metadata()
        
        # Add hyperparameter search results if available
        if hasattr(self, 'model') and hasattr(self.model, 'best_params_'):
            metadata.update({
                "hyperparameter_optimization": {
                    "best_params": self.model.best_params_,
                    "best_score": float(self.model.best_score_),
                    "n_trials": self.model.n_iter_,
                    "cv_results": {
                        "mean_test_score": float(np.mean(self.model.cv_results_['test_score'])),
                        "std_test_score": float(np.std(self.model.cv_results_['test_score']))
                    }
                }
            })
        
        return metadata