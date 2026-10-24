from pathlib import Path
from typing import Dict, List, Type, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_log_error
import uuid
from .submission_manager import SegmentSubmissionManager

class SegmentedModelTrainer:
    """Trains and manages models for different data segments"""
    
    @staticmethod
    def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Logarithmic Error
        
        Args:
            y_true: Ground truth target values
            y_pred: Predicted target values
            
        Returns:
            float: RMSLE score
        """
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    
    def __init__(
        self,
        base_model_class: Type[BaseEstimator],
        feature_segments: Dict[str, List[tuple]],
        output_dir: Path,
        model_params: Optional[Dict[str, Any]] = None,
        test_mode: bool = False
    ):
        """Initialize trainer with test mode awareness"""
        self.base_model_class = base_model_class
        self.feature_segments = feature_segments
        self.output_dir = Path(output_dir)
        self.models: Dict[str, BaseEstimator] = {}
        self._model_params: Dict[str, Any] = model_params or {}
        self.test_mode = test_mode
        self.segment_scores = {}
        self.is_trained = False
        
        if not test_mode:
            # Only create output directories for non-test runs
            self.models_dir = self.output_dir / 'models'
            self.predictions_dir = self.output_dir / 'predictions'
            for dir_path in [self.models_dir, self.predictions_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize submission manager with correct parameter
        self.submission_manager = SegmentSubmissionManager(
            predictions_dir=self.predictions_dir
        )
    
    @property
    def model_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return self._model_params
    
    @model_params.setter
    def model_params(self, params: Dict[str, Any]) -> None:
        """Set model parameters with validation"""
        if not isinstance(params, dict):
            raise TypeError("Model parameters must be a dictionary")
        self._model_params = params.copy()
    
    def update_model_params(self, params: Dict[str, Any]) -> None:
        """Update model parameters for quick testing or parameter tuning"""
        self.model_params = params  # Uses the setter for validation
        print(f"Updated model parameters: {self.model_params}")
        
    def create_model(self) -> BaseEstimator:
        """Create a new model instance with current parameters"""
        return self.base_model_class(**self.model_params)
    
    def train_segment_models(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target: str = 'PremiumAmount'
    ) -> Dict[str, Dict[str, float]]:
        """Train models for each segment and return performance metrics"""
        segment_scores = {}
        
        # Verify target column exists
        if target not in train_data.columns:
            raise ValueError(f"Target column '{target}' not found in training data. "
                            f"Available columns: {train_data.columns.tolist()}")
        
        # Extract target values, handling both pandas and cuDF
        try:
            if hasattr(train_data, 'to_pandas'):
                # GPU DataFrame
                train_df = train_data.to_pandas()
                y_train = train_df[target].values
            else:
                # CPU DataFrame
                y_train = train_data[target].values
        except KeyError as e:
            raise KeyError(f"Failed to extract target column '{target}'. "
                          f"Available columns: {train_data.columns.tolist()}") from e
        
        # Identify columns to drop
        drop_cols = []
        for col in ['id', target]:
            if col in train_data.columns:
                drop_cols.append(col)
        
        # Add categorical columns
        categorical_cols = [col for col in train_data.columns if col.endswith('_Segment')]
        drop_cols.extend(categorical_cols)
        
        for feature, segments in self.feature_segments.items():
            print(f"\nTraining models for {feature} segments...")
            
            for segment_range in segments:
                segment_name = f"{feature}_{segment_range[0]}_{segment_range[1]}"
                
                # Get segment mask and data
                try:
                    if hasattr(train_data, 'to_pandas'):
                        # Handle GPU DataFrame
                        feature_data = train_data[feature].to_pandas()
                        train_mask = (
                            (feature_data >= segment_range[0]) & 
                            (feature_data < segment_range[1])
                        ).values
                    else:
                        train_mask = (
                            (train_data[feature] >= segment_range[0]) & 
                            (train_data[feature] < segment_range[1])
                        ).values
                except KeyError as e:
                    raise KeyError(f"Failed to create segment mask for feature '{feature}'. "
                                 f"Available columns: {train_data.columns.tolist()}") from e
                
                if train_mask.sum() < 100:
                    print(f"Skipping {segment_name} - insufficient samples: {train_mask.sum()}")
                    continue
                
                print(f"Training {segment_name} with {train_mask.sum()} samples...")
                
                # Prepare training data
                X_train = train_data[train_mask].copy()
                y_segment = y_train[train_mask]
                
                # Drop columns safely
                cols_to_drop = [col for col in drop_cols if col in X_train.columns]
                X_train = X_train.drop(columns=cols_to_drop)
                
                # Create and train model
                model = self.create_model()
                try:
                    import cudf
                    if isinstance(X_train, cudf.DataFrame):
                        y_segment = cudf.Series(y_segment)
                except ImportError:
                    pass
                
                model = self._train_segment(X_train, y_segment, model)
                self.models[segment_name] = model
                
                # Evaluate segment
                segment_scores[segment_name] = self._evaluate_segment(
                    model=model,
                    segment_name=segment_name,
                    segment_range=segment_range,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    feature=feature,
                    target=target,
                    drop_cols=cols_to_drop
                )
        
        self.segment_scores = segment_scores
        self.is_trained = True
        return segment_scores

    def _evaluate_segment(
        self,
        model: BaseEstimator,
        segment_name: str,
        segment_range: tuple,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature: str,
        target: str,
        drop_cols: List[str]
    ) -> Dict[str, Any]:
        """Evaluate model on a segment with GPU support"""
        scores = {
            'segment_name': segment_name,
            'feature': feature,
            'segment_range': segment_range,
            'train_samples': 0,
            'val_samples': 0,
            'train_score': None,
            'val_score': None,
            'timestamp': datetime.now().isoformat()
        }
        
        def safe_rmsle(y_true, y_pred) -> float:
            """Calculate RMSLE with protection against negative values"""
            # Convert GPU arrays to numpy if needed
            if hasattr(y_true, 'to_numpy'):
                y_true = y_true.to_numpy()
            if hasattr(y_pred, 'to_numpy'):
                y_pred = y_pred.to_numpy()
                
            # Ensure arrays are numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            
            # Clip values
            min_val = max(1e-6, float(y_true.min()))
            y_pred_clipped = np.clip(y_pred, min_val, None)
            y_true_clipped = np.clip(y_true, min_val, None)
            
            return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))
        
        for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            # Create segment mask
            mask = (data[feature] >= segment_range[0]) & (data[feature] < segment_range[1])
            if hasattr(mask, 'to_numpy'):
                mask = mask.to_numpy()
                
            segment_data = data[mask].copy()
            
            if len(segment_data) > 0:
                # Prepare features
                X = segment_data.drop(columns=drop_cols)
                y = segment_data[target]
                
                # Convert target to numpy if it's a GPU array
                if hasattr(y, 'to_numpy'):
                    y = y.to_numpy()
                
                # Train or predict
                if name == 'train':
                    model.fit(X, y)
                
                predictions = model.predict(X)
                
                # Calculate scores for train and val
                if name != 'test' and target in segment_data.columns:
                    score = safe_rmsle(y, predictions)
                    scores[f'{name}_score'] = float(score)
                    scores[f'{name}_samples'] = int(len(segment_data))
        
        return scores

    def _train_segment(self, X: pd.DataFrame, y: pd.Series, model: BaseEstimator) -> BaseEstimator:
        """Train model on segment data with proper device placement"""
        if hasattr(model, 'predictor') and model.predictor == 'gpu_predictor':
            # Ensure data is on GPU
            try:
                import cudf
                if not isinstance(X, cudf.DataFrame):
                    X = cudf.DataFrame(X)
                if not isinstance(y, cudf.Series):
                    y = cudf.Series(y)
            except ImportError:
                pass
        
        model.fit(X, y)
        return model

    def get_segment_scores(self) -> Dict[str, float]:
        """Get RMSLE scores for each segment"""
        if not self.is_trained or not hasattr(self, 'segment_scores'):
            return {}
        return {
            segment: scores.get('val_score', float('inf')) 
            for segment, scores in self.segment_scores.items()
            if scores.get('val_score') is not None
        }

    def get_mean_validation_score(self) -> float:
        """Get mean RMSLE across all segments"""
        scores = [
            score['val_score'] 
            for score in self.segment_scores.values() 
            if score.get('val_score') is not None
        ]
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def get_weighted_validation_score(self) -> float:
        """Get sample-weighted mean RMSLE across segments"""
        scores = [
            (score['val_score'], score['val_samples'])
            for score in self.segment_scores.values()
            if score.get('val_score') is not None and score.get('val_samples', 0) > 0
        ]
        
        if not scores:
            return 0.0
        
        total_samples = sum(samples for _, samples in scores)
        weighted_score = sum(
            score * samples / total_samples
            for score, samples in scores
        )
        return float(weighted_score)

    def _update_segment_score(self, segment: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Update RMSLE score for a segment"""
        from sklearn.metrics import mean_squared_log_error
        try:
            rmsle = float(np.sqrt(mean_squared_log_error(y_true, y_pred)))
            self.segment_scores[segment] = rmsle
            if not hasattr(self, 'segment_samples'):
                self.segment_samples = {}
            self.segment_samples[segment] = len(y_true)
        except Exception as e:
            print(f"Warning: Could not calculate RMSLE for segment {segment}: {e}")
            self.segment_scores[segment] = float('inf')

    def train(self, X: pd.DataFrame, y: pd.Series, segments: Dict[str, List[tuple]]) -> None:
        """Train models for each segment with score tracking"""
        self.models = {}
        self.segment_scores = {}
        self.segment_samples = {}
        
        for feature, ranges in segments.items():
            for i, (min_val, max_val) in enumerate(ranges):
                segment = f"{feature}_{i}"
                mask = (X[feature] >= min_val) & (X[feature] < max_val)
                
                if mask.sum() > 0:
                    X_segment = X[mask]
                    y_segment = y[mask]
                    
                    # Create and train model
                    model = self.base_model_class()
                    model.fit(X_segment, y_segment)
                    
                    # Save model and update scores
                    self.models[segment] = model
                    y_pred = model.predict(X_segment)
                    self._update_segment_score(segment, y_segment, y_pred)
                    
                    print(f"Trained {segment}: {len(X_segment)} samples, "
                          f"RMSLE: {self.segment_scores[segment]:.4f}")
        
        self.is_trained = True
