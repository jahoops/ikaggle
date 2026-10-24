from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_log_error
from .segment_model_trainer import SegmentedModelTrainer  # Changed to relative import
from segment_data.optimal_segments import OPTIMAL_SEGMENTS  # Add this import
import json  # Add this import
from datetime import datetime  # Add this import
import os  # Add this import
from time import time  # Add this import
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold

class BaseSegmentExperiment(ABC):
    """Base class for all segment-specific experiments"""
    
    def __init__(self, output_dir: Path, test_mode: bool = False):
        """Initialize experiment with proper output structure"""
        # Create base output directory structure
        self.base_output = Path('/home/USER/Projects/ikaggle/segment_output')
        
        # Create timestamped directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = self.__class__.__name__.lower()
        run_type = '_test' if test_mode else '_prod'
        
        # Full output path structure:
        # /home/USER/Projects/ikaggle/segment_output/{experiment_name}/{timestamp}_{run_type}/
        self.output_dir = (self.base_output / experiment_name / 
                          f"{timestamp}{run_type}")
        
        # Create specific subdirectories
        self.models_dir = self.output_dir / 'models'
        self.predictions_dir = self.output_dir / 'predictions'
        self.metrics_dir = self.output_dir / 'metrics'
        self.plots_dir = self.output_dir / 'plots'
        
        # Create all directories
        for directory in [self.output_dir, self.models_dir, 
                         self.predictions_dir, self.metrics_dir,
                         self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize other attributes
        self.name = self.__class__.__name__
        self.test_mode = test_mode
        self.optimal_segments = OPTIMAL_SEGMENTS
        self.scaler = RobustScaler()
        self.feature_cols: List[str] = []
        self.experiment_info = {
            'name': self.name,
            'test_mode': test_mode,
            'timestamp': pd.Timestamp.now().isoformat(),
            'segment_scores': {},
            'overall_scores': {},
            'feature_importance': {},
            'segment_stats': {},
            'notes': {
                'hypothesis': '',
                'methodology': '',
                'observations': '',
                'next_steps': '',
                'timestamp': '',
                'author': ''
            },
            'performance': {
                'duration': 0,
                'test_duration': 0
            }
        }
        self.categorical_features = [
            'Gender', 'MaritalStatus', 'EducationLevel',
            'Occupation', 'Location', 'PolicyType',
            'CustomerFeedback', 'SmokingStatus',
            'ExerciseFrequency', 'PropertyType'
        ]
        self.setup_directories()
        self.model_trainer = None
    
    def setup_directories(self) -> None:
        """Create necessary output directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
        if not self.test_mode:
            # Only create detailed output dirs for non-test runs
            self.model_dir = self.output_dir / 'models'
            self.prediction_dir = self.output_dir / 'predictions'
            self.analysis_dir = self.output_dir / 'analysis'
        
            for dir_path in [self.model_dir, self.prediction_dir, self.analysis_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Logarithmic Error"""
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    def get_segment_boundaries(self, data: pd.DataFrame, feature: str, n_segments: int = 6) -> List[Tuple[float, float]]:
        """Calculate segment boundaries using quantiles"""
        quantiles = np.linspace(0, 1, n_segments + 1)
        boundaries = data[feature].quantile(quantiles).values
        return [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]
    
    def encode_categoricals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables consistently"""
        data = data.copy()
        for feature in self.categorical_features:
            if feature in data.columns:
                dummies = pd.get_dummies(data[feature], prefix=feature)
                data = pd.concat([data, dummies], axis=1)
                data.drop(columns=[feature], inplace=True)
        return data
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values consistently across all experiments"""
        data = data.copy()
        # Fill numeric columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        # Fill categorical columns with 'Unknown'
        cat_cols = [col for col in data.columns if col in self.categorical_features]
        data[cat_cols] = data[cat_cols].fillna('Unknown')
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing common to all segment experiments"""
        # Create working copy
        result = data.copy()
        
        # Basic data validation
        if 'PremiumAmount' not in result.columns and not self.test_mode:
            raise ValueError(f"Target column 'PremiumAmount' not found. "
                            f"Available columns: {result.columns.tolist()}")
        
        # Handle missing values
        if result.isnull().any().any():
            for col in result.columns:
                if result[col].isnull().any():
                    if pd.api.types.is_numeric_dtype(result[col]):
                        result[col] = result[col].fillna(result[col].mean())
                    else:
                        result[col] = result[col].fillna('missing')
        
        return result
    
    @abstractmethod
    def create_model(self) -> BaseEstimator:
        """Create and return the model to be used for this experiment"""
        pass
    
    @abstractmethod
    def get_feature_segments(self, data: pd.DataFrame) -> Dict[str, List[tuple]]:
        """Get feature segments - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_quick_test_params(self) -> Dict[str, Any]:
        """Return parameters for quick test mode
        
        Must define a minimal set of parameters that allows quick testing
        of the full pipeline while sacrificing model performance.
        
        Returns:
            Dict with model parameters optimized for speed over accuracy
        """
        pass
    
    def transform_income(self, income: float) -> float:
        """Standard income transformation"""
        clusters = [5356.18, 22594.06, 43825.62, 77021.06, 122424.68]
        return next((i / len(clusters) for i, cluster in enumerate(clusters) 
                    if income < cluster), 1.0)
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data"""
        required_cols = ['id', 'PremiumAmount', 'AnnualIncome']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def run(self, train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        """Run experiment with proper validation"""
        # Remove notes validation for now - we'll add them at the end
        if train_data is None or val_data is None or test_data is None:
            raise ValueError("All datasets (train, validation, test) are required")
        
        # Continue with experiment
        self.train(train_data)
        self.validate(val_data)
        self.test(test_data)
    
    def get_segment_label(self, feature: str, value: float) -> str:
        """Get the segment label for a given feature value"""
        segments = self.optimal_segments[feature]
        boundaries = segments['boundaries']
        labels = segments['labels']
        
        for i in range(len(labels)):
            if value >= boundaries[i] and value < boundaries[i + 1]:
                return labels[i]
        return labels[-1]  # Return last label if value >= max boundary
    
    def segment_feature(self, data: pd.DataFrame, feature: str) -> pd.Series:
        """Add segment labels for a feature"""
        return data[feature].apply(lambda x: self.get_segment_label(feature, x))
    
    def get_model_path(self, segment: Optional[str] = None) -> Path:
        """Get path for saving/loading model"""
        base_name = f"{self.name}_{'test' if self.test_mode else 'prod'}"
        if segment:
            return self.model_dir / f"{base_name}_{segment}.model"
        return self.model_dir / f"{base_name}.model"
    
    def get_prediction_path(self, dataset: str) -> Path:
        """Get path for saving predictions"""
        return self.prediction_dir / f"{self.name}_{dataset}_pred.csv"
    
    def get_analysis_path(self, name: str) -> Path:
        """Get path for saving analysis results"""
        return self.analysis_dir / f"{self.name}_{name}.csv"
    
    def save_experiment_info(self) -> None:
        """Save experiment metadata and scores"""
        info_path = self.output_dir / 'experiment_info.json'
        with open(info_path, 'w') as f:
            json.dump(self.experiment_info, f, indent=2)

    def update_segment_score(self, segment: str, scores: Dict[str, float]) -> None:
        """Update scores for a specific segment"""
        self.experiment_info['segment_scores'][segment] = {
            'rmsle': scores['rmsle'],
            'sample_count': scores['samples'],
            'validation_score': scores['val_score'],
            'segment_range': scores['range']
        }
        self.save_experiment_info()

    def update_overall_scores(self, scores: Dict[str, float]) -> None:
        """Update overall experiment scores"""
        self.experiment_info['overall_scores'].update(scores)
        self.save_experiment_info()

    def update_feature_importance(self, feature_importance: pd.DataFrame) -> None:
        """Update feature importance scores"""
        self.experiment_info['feature_importance'] = feature_importance.to_dict()
        self.save_experiment_info()

    def update_segment_stats(self, feature: str, stats: pd.DataFrame) -> None:
        """Update statistics for each segment"""
        self.experiment_info['segment_stats'][feature] = stats.to_dict()
        self.save_experiment_info()

    def add_experiment_notes(
        self,
        hypothesis: str,
        methodology: str,
        observations: str,
        next_steps: str,
        author: Optional[str] = None
    ) -> None:
        """Add structured notes to experiment"""
        self.experiment_info['notes'].update({
            'hypothesis': hypothesis,
            'methodology': methodology,
            'observations': observations,
            'next_steps': next_steps,
            'timestamp': datetime.now().isoformat(),
            'author': author or os.getenv('USER', 'unknown')
        })
        self.save_experiment_info()

    def get_output_paths(self) -> Dict[str, Path]:
        """Get dictionary of all output paths"""
        return {
            'base': self.output_dir,
            'models': self.models_dir,
            'predictions': self.predictions_dir,
            'metrics': self.metrics_dir,
            'plots': self.plots_dir
        }

    def save_predictions(self, predictions: pd.DataFrame, dataset_type: str) -> None:
        """Save predictions to appropriate location"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_type}_predictions_{timestamp}.csv"
        output_path = self.predictions_dir / filename
        predictions.to_csv(output_path, index=False)
        print(f"Saved {dataset_type} predictions to: {output_path}")

    def save_metrics(self, metrics: Dict[str, float], dataset_type: str) -> None:
        """Save metrics to a JSON file
        
        Args:
            metrics: Dictionary of metric names and values
            dataset_type: Type of dataset ('train', 'validation', or 'test')
        """
        metrics_dir = self.output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        
        metrics_path = metrics_dir / f'{dataset_type}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def save_predictions_with_ids(self, ids: pd.Series, predictions: np.ndarray, dataset_type: str) -> None:
        """Save predictions with IDs to standardized location"""
        # Create predictions directory if it doesn't exist
        predictions_dir = self.output_dir
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standardized filename
        output_path = predictions_dir / f'{dataset_type}_predictions.csv'
        print(f"Saving predictions to: {output_path}")
        
        # Save predictions
        pd.DataFrame({
            'id': ids,
            'PremiumAmount': predictions
        }).to_csv(output_path, index=False)

    def get_feature_segments(self, data: pd.DataFrame) -> Dict[str, List[tuple]]:
        """Create optimal segments for features based on data distribution"""
        segments = {}
        min_samples = 100
        
        # Default segment counts per feature
        n_segments = {
            'Age': 10,
            'AnnualIncome': 10,
            'HealthScore': 9,
            'CreditScore': 9,
            'NumberofDependents': 5
        }
        
        for feature in self.numeric_features:
            if feature in data.columns:
                feature_data = data[feature].dropna()
                segments[feature] = self._create_quantile_segments(
                    feature_data,
                    n_segments.get(feature, 5),
                    min_samples
                )
                
        return segments

    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Collect GPU metrics if available"""
        metrics = {
            "gpu_enabled": False,
            "device_name": None,
            "memory_allocated": None,
            "compute_capability": None
        }
        
        if self.use_gpu:
            try:
                import cupy as cp
                dev = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                free_memory, total_memory = dev.mem_info
                
                metrics.update({
                    "gpu_enabled": True,
                    "device_name": props['name'].decode('utf-8'),
                    "memory_allocated": float((total_memory - free_memory) / (1024**2)),
                    "memory_available": float(free_memory / (1024**2)),
                    "total_memory": float(total_memory / (1024**2)),
                    "compute_capability": f"{props['major']}.{props['minor']}"
                })
            except Exception as e:
                print(f"Warning: Could not get GPU metrics: {e}")
        
        return metrics

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_type: str) -> Dict[str, float]:
        """Calculate standard regression metrics with proper clipping
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            dataset_type: Type of evaluation (val/test)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_log_error, mean_squared_error
        
        # Ensure non-negative values
        y_pred = np.clip(y_pred, 1e-6, None)
        y_true = np.clip(y_true, 1e-6, None)
        
        metrics = {
            f'{dataset_type}_rmsle': float(np.sqrt(mean_squared_log_error(y_true, y_pred))),
            f'{dataset_type}_rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            f'{dataset_type}_samples': len(y_true)
        }
        
        print(f"\n{dataset_type.capitalize()} metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics

    def optimize_hyperparameters(self, train_data: pd.DataFrame) -> BaseEstimator:
        """Optimize model hyperparameters using RandomizedSearchCV
        
        Args:
            train_data: Training data with features and target
            
        Returns:
            Optimized model instance
        """
        # Prepare data
        X = train_data[self.feature_columns]
        y = train_data['PremiumAmount']
        
        # Get param distributions based on mode
        param_distributions = (
            self.get_quick_param_distributions() 
            if self.test_mode 
            else self.get_param_distributions()
        )
        
        # Setup cross-validation
        cv = RepeatedKFold(
            n_splits=2 if self.test_mode else 5,
            n_repeats=1 if self.test_mode else 2,
            random_state=42
        )
        
        # Create and run search
        search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=5 if self.test_mode else 30,
            scoring='neg_mean_squared_log_error',
            cv=cv,
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        
        search.fit(X, y)
        
        # Log best parameters
        print(f"\nBest parameters found:")
        for param, value in search.best_params_.items():
            print(f"{param}: {value}")
        
        return search.best_estimator_

    def _create_quantile_segments(self, feature_data: pd.Series, n_segments: int, min_samples: int) -> List[tuple]:
        """Create segments based on quantiles with minimum sample size constraint
        
        Args:
            feature_data: Series containing feature values
            n_segments: Target number of segments
            min_samples: Minimum samples required per segment
            
        Returns:
            List of tuples containing (lower_bound, upper_bound) for each segment
        """
        # Check if we have enough samples for requested segments
        if len(feature_data) < min_samples * n_segments:
            n_segments = max(2, len(feature_data) // min_samples)
            print(f"Warning: Reducing segments to {n_segments} due to sample size constraint")
        
        # Calculate quantiles
        quantiles = np.linspace(0, 1, n_segments + 1)
        boundaries = feature_data.quantile(quantiles).values
        
        # Create segments
        segments = []
        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            upper = boundaries[i + 1]
            
            # Adjust upper bound of last segment to include maximum value
            if i == len(boundaries) - 2:
                upper = upper * 1.001  # Add small buffer
                
            segments.append((float(lower), float(upper)))
        
        return segments