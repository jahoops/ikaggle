from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xgb
from datetime import datetime
from .base_segment_experiment import BaseSegmentExperiment
from .segment_model_trainer import SegmentedModelTrainer

class XGBoostSegmentExperiment(BaseSegmentExperiment):
    """XGBoost-based segment experiment with GPU support"""
    
    def __init__(self, output_dir: Path, test_mode: bool = False):
        # Define class attributes before super().__init__
        self.numeric_features = [
            'Age',
            'AnnualIncome', 
            'HealthScore',
            'CreditScore',
            'NumberofDependents'
        ]
        self.feature_columns = self.numeric_features.copy()
        self.model = None
        self.model_trainer = None
        
        # Call super().__init__ after defining attributes
        super().__init__(output_dir, test_mode)
        
        # Initialize GPU support after super().__init__
        self.use_gpu = self._check_gpu_available()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for XGBoost"""
        try:
            # Create tiny dataset to test GPU
            import cudf
            return True
        except ImportError:
            print("GPU support not available, using CPU")
            return False
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters based on mode"""
        base_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Changed from gpu_hist
            'device': 'cuda' if self.use_gpu else 'cpu',  # New device parameter
            'random_state': 42
        }
        
        if not self.test_mode:
            # Production parameters
            base_params.update({
                'n_estimators': 500,
                'max_depth': 8,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            })
            
        return base_params
    
    def get_quick_test_params(self) -> Dict[str, Any]:
        """Return parameters optimized for quick testing"""
        return {
            'n_estimators': 50,
            'max_depth': 4,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Changed from gpu_hist
            'device': 'cuda' if self.use_gpu else 'cpu',  # New device parameter
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'early_stopping_rounds': 5
        }
    
    def create_model(self) -> xgb.XGBRegressor:
        """Create a new XGBoost model instance with appropriate parameters"""
        if self.test_mode:
            params = self.get_quick_test_params()
        else:
            params = self.get_model_params()
        return xgb.XGBRegressor(**params)
    
    def train(self, train_data: pd.DataFrame) -> None:
        """Train XGBoost model with hyperparameter optimization"""
        # Optimize hyperparameters
        self.model = self.optimize_hyperparameters(train_data)
        
        # Save optimized model
        model_path = self.models_dir / f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.model.save_model(str(model_path))

    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model and use base class metrics calculation"""
        if self.model is None:
            raise RuntimeError("Model must be trained before validation")

        # Prepare validation data
        X_val = val_data[self.feature_columns]
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
        X_test = test_data[self.feature_columns]
        y_test = test_data['PremiumAmount'].values
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        
        # Use base class metrics calculation
        return self.calculate_metrics(y_test, y_pred, 'test')

    def initialize_model_trainer(self, train_data: pd.DataFrame) -> None:
        """Initialize model trainer with data-dependent segments"""
        self.model_trainer = SegmentedModelTrainer(
            base_model_class=self.create_model().__class__,
            feature_segments=self.get_feature_segments(train_data),
            output_dir=self.output_dir
        )

    def save_model(self, model: xgb.XGBRegressor, segment: Optional[str] = None) -> None:
        """Save XGBoost model to appropriate path"""
        model_path = self.get_model_path(segment)
        model.save_model(str(model_path))

    def load_model(self, segment: Optional[str] = None) -> xgb.XGBRegressor:
        """Load XGBoost model from appropriate path"""
        model_path = self.get_model_path(segment)
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")
        model = self.create_model()
        model.load_model(str(model_path))
        return model

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """XGBoost-specific preprocessing with feature engineering"""
        # First apply base preprocessing
        data = super().preprocess_data(data)
        result = data.copy()
        
        # Ensure target is non-negative if present
        if 'PremiumAmount' in result.columns:
            result['PremiumAmount'] = result['PremiumAmount'].clip(lower=1e-6)
        
        # XGBoost-specific feature engineering
        for feature in self.numeric_features:
            if feature in result.columns:
                # Ensure feature values are non-negative for log transforms
                if feature in ['AnnualIncome', 'HealthScore']:
                    result[feature] = result[feature].clip(lower=1e-6)
                    result[f'{feature}_Log'] = np.log1p(result[feature])
                # Polynomial features for specific predictors
                if feature in ['CreditScore', 'Age']:
                    result[f'{feature}_Squared'] = result[feature] ** 2
                    result[feature] = result[feature].clip(lower=1e-6)
        # Move data to GPU if available and preserve target
        if self.use_gpu:
            try:
                import cudf
                # Keep target column separated
                target_value = None
                if 'PremiumAmount' in result.columns:
                    target_value = result['PremiumAmount'].copy()
                
                # Convert features to GPU
                feature_cols = [col for col in result.columns 
                              if col not in ['id', 'PremiumAmount']]
                result_gpu = cudf.DataFrame(result[feature_cols])
                # Add target back to GPU DataFrame
                if target_value is not None:
                    result_gpu['PremiumAmount'] = target_value
                return result_gpu
            except (ImportError, Exception) as e:
                print(f"Warning: GPU data conversion failed ({str(e)}), using CPU")
                self.use_gpu = False
        return result

    def _ensure_device_match(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure data is on the correct device for XGBoost"""
        if self.use_gpu:
            try:
                import cudf
                if not isinstance(X, cudf.DataFrame):
                    return cudf.DataFrame(X)
            except ImportError:
                self.use_gpu = False
        return X

    def run(self, train_data: pd.DataFrame, 
            val_data: pd.DataFrame, 
            test_data: pd.DataFrame) -> None:
        """Run experiment with proper initialization"""
        if self.model_trainer is None:
            self.initialize_model_trainer(train_data)
        super().run(train_data, val_data, test_data)

    def get_experiment_metadata(self) -> Dict:
        """Return metadata for the experiment including GPU metrics and training scores"""
        metadata = {
            "name": "XGBoostSegmentExperiment",
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
                
                # Get device properties using the correct CuPy API
                try:
                    props = cp.cuda.runtime.getDeviceProperties(dev.id)
                    device_name = props['name'].decode('utf-8')
                    compute_major = props['major']
                    compute_minor = props['minor']
                    
                    # Get memory information - use mem_info as a property
                    free_memory, total_memory = dev.mem_info  # Access as property
                    memory_used = total_memory - free_memory
                    
                    metadata["hardware_config"].update({
                        "device_name": device_name,
                        "memory_allocated": float(memory_used / (1024**2)),  # MB
                        "memory_available": float(free_memory / (1024**2)),  # MB
                        "total_memory": float(total_memory / (1024**2)),  # MB
                        "compute_capability": f"{compute_major}.{compute_minor}"
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
                
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not get GPU metrics: {e}")
                
        # Add model information and training metrics
        if hasattr(self, 'model_trainer') and self.model_trainer is not None:
            metadata.update({
                "model_info": {
                    "n_segments": len(self.model_trainer.models) if hasattr(self.model_trainer, 'models') else 0,
                    "training_complete": hasattr(self.model_trainer, 'is_trained') and self.model_trainer.is_trained,
                    "segment_scores": getattr(self.model_trainer, 'segment_scores', {})
                }
            })
        
        return metadata

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained models"""
        importance_dict = {}
        
        if hasattr(self, 'model_trainer') and self.model_trainer is not None:
            for segment, model in self.model_trainer.models.items():
                if hasattr(model, 'feature_importances_'):
                    for feat, imp in zip(self.numeric_features, model.feature_importances_):
                        if feat not in importance_dict:
                            importance_dict[feat] = []
                        importance_dict[feat].append(float(imp))
            # Average importance across segments
            return {
                feat: float(np.mean(scores)) 
                for feat, scores in importance_dict.items()
            }
            
        return {}

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions ensuring non-negative values"""
        X = data[self.feature_columns]
        
        # Move data to GPU if model is using GPU
        if self.use_gpu:
            try:
                import cudf
                X = cudf.DataFrame(X)
            except ImportError:
                print("Warning: GPU support not available for prediction")
                self.use_gpu = False
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Move predictions back to CPU if needed
        if isinstance(predictions, cudf.Series):
            predictions = predictions.values.get()
        
        # Ensure non-negative predictions
        predictions = np.clip(predictions, 1e-6, None)
        
        return pd.DataFrame({'PremiumAmount': predictions})

    def get_param_distributions(self) -> Dict[str, List[Any]]:
        """Define full hyperparameter search space"""
        return {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }

    def get_quick_param_distributions(self) -> Dict[str, List[Any]]:
        """Define minimal parameter search space for testing"""
        return {
            'n_estimators': [100],
            'max_depth': [4],
            'learning_rate': [0.1],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'min_child_weight': [1]
        }