import logging
import time
from pathlib import Path
import sys
import itertools
import json
from datetime import datetime
import platform
import joblib
import os
from collections import defaultdict

# Ensure experiments module is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from cuml.svm import SVR, LinearSVR
from cuml.preprocessing import RobustScaler
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from experiments.data_manager import data_manager
from experiments.base_experiment import BaseExperiment
from experiments import feature_cols  # Import feature_cols

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def safe_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error (RMSLE) safely."""
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    assert len(y_true) == len(y_pred)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    log_true = np.log1p(cp.asnumpy(y_true))
    log_pred = np.log1p(y_pred.get())
    return np.sqrt(np.mean((log_true - log_pred) ** 2))

class MixedSVR:
    def __init__(self):
        # Create multiple SVR models with different kernels
        self.svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        self.svr_lin = SVR(kernel='linear', C=100, epsilon=.1)
        self.svr_poly = SVR(kernel='poly', C=100, degree=2, epsilon=.1)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit each model
        self.svr_rbf.fit(X_scaled, y)
        self.svr_lin.fit(X_scaled, y)
        self.svr_poly.fit(X_scaled, y)
        return self

    def predict(self, X):
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        pred_rbf = self.svr_rbf.predict(X_scaled)
        pred_lin = self.svr_lin.predict(X_scaled)
        pred_poly = self.svr_poly.predict(X_scaled)
        
        # Combine predictions (simple average)
        return (pred_rbf + pred_lin + pred_poly) / 3

class SVRExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "SVR"
    
    def __init__(self, output_dir, test_mode=False):
        super().__init__(output_dir, test_mode)
        
        # Create run-specific directory
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging for both file and console
        log_file = self.run_dir / f"{self.name}_{self.run_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(message)s')
        )
        
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        # Log system details
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"CUDA available: {cp.cuda.is_available()}")
        if cp.cuda.is_available():
            self.logger.info(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
        
        self.feature_weights = {
            'HealthScore': 0.5523, 'AnnualIncome': 0.2245,
            'CreditScore': 0.0800, 'Age': 0.0085,
            'SignupYear': 0.0023, 'PreviousClaims': 0.0021,
            'NumberofDependents': 0.0018, 'Location_Rural': 0.0015,
            'VehicleAge': 0.0015, 'InsuranceDuration': 0.0015
        }
        if not cp.cuda.is_available():
            raise RuntimeError("GPU required but not available")
    
    def create_model(self, C=1.0, epsilon=0.1, kernel='linear'):
        if kernel == 'linear':
            return LinearSVR(
                C=C,
                epsilon=epsilon,
                max_iter=100000,
                tol=1e-5
            )
        else:
            return SVR(
                C=C,
                epsilon=epsilon,
                kernel=kernel,
                max_iter=100000,
                tol=1e-5
            )

    def save_model(self, model, params, score):
        model_dir = self.output_dir / self.name / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "best_model.joblib"
        joblib.dump(model, model_path)
        
        metadata = {
            "params": params,
            "score": float(score),
            "timestamp": self.format_time(time.time())
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save the scaler
        scaler_path = model_dir / "robust_scaler.joblib"
        joblib.dump(self.X_scaler, scaler_path)
        self.logger.info(f"Saved scaler to {scaler_path}")

        # Save the target scaler
        target_scaler_path = model_dir / "target_scaler.joblib"
        joblib.dump(self.y_scaler, target_scaler_path)
        self.logger.info(f"Saved target scaler to {target_scaler_path}")

        # Save the feature columns
        feature_cols_path = model_dir / "feature_cols.joblib"
        joblib.dump(self.feature_cols, feature_cols_path)
        self.logger.info(f"Saved feature columns to {feature_cols_path}")
        
        self.logger.info(f"Saved model to {model_path}")

    def load_model(self):
        model_dir = self.output_dir / self.name / "model"
        model_path = model_dir / "best_model.joblib"
        if not model_path.exists():
            return None, None
        model = joblib.load(model_path)
        try:
            with open(model_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
        except FileNotFoundError:
            metadata = None
        return model, metadata

    def evaluate_model(self, model, X, y):
        """Evaluate model with proper GPU/CPU array handling."""
        # Handle different input types
        if isinstance(X, cp.ndarray):
            X = cudf.DataFrame(X)  # Convert CuPy array to cuDF DataFrame
        elif isinstance(X, np.ndarray):
            X = cudf.DataFrame.from_pandas(pd.DataFrame(X))
        
        if isinstance(y, (cp.ndarray, np.ndarray)):
            y = cudf.Series(y)  # Convert array to cuDF Series
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert predictions and targets to numpy for scoring
        if isinstance(predictions, (cudf.Series, cp.ndarray)):
            predictions = predictions.get() if hasattr(predictions, 'get') else predictions.values
        if isinstance(y, (cudf.Series, cp.ndarray)):
            y = y.get() if hasattr(y, 'get') else y.values
        
        predictions = predictions.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        # Inverse transform using CPU scalers
        y_original = self.y_scaler.inverse_transform(y).flatten()
        predictions_original = self.y_scaler.inverse_transform(predictions).flatten()
        
        score = safe_rmsle(y_original, predictions_original)
        return -float(score)

    def convert_to_list(self, arr):
        """Helper function to convert arrays to Python lists."""
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        if isinstance(arr, (list, tuple)):
            return list(arr)
        if hasattr(arr, 'to_arrow'):
            return arr.to_arrow().to_pylist()
        if hasattr(arr, 'values'):  # For pandas/cudf Series
            return arr.values.tolist()
        return list(arr)

    def run(self):
        start_time = time.time()
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / self.name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        best_score = float('-inf')  # Initialize to negative infinity
        best_model = None
        best_params = None  # To record the parameters of the best model
        
        param_grid = {
            'C': [1000.0, 5000.0, 10000.0, 50000.0, 100000.0],  # Trending towards higher C
            'epsilon': [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01],  # More granular small epsilon
            'kernel': ['linear']
        }
        total_iterations = len(list(itertools.product(*param_grid.values())))
        current_iteration = 0

        data = data_manager.finaltrain.copy()
        print("\nColumns before feature selection:", sorted(data.columns.tolist()))
        
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']
        ids = data['id']

        # Use feature columns from experiments module and ensure uniqueness
        X = X[feature_cols]
        
        # Check for and remove any duplicate columns
        dupe_cols = X.columns[X.columns.duplicated()].tolist()
        if (dupe_cols):
            self.logger.warning(f"Found duplicate columns: {dupe_cols}")
            X = X.loc[:, ~X.columns.duplicated()]
        
        print("\nColumns after deduplication:", sorted(X.columns.tolist()))
        
        # Convert to GPU tensors while preserving ID relationship
        X_gpu = cudf.DataFrame.from_pandas(X)
        X_gpu['__id__'] = ids  # Add ID column to features
        y_gpu = cudf.Series(y)
        
        # Initialize and fit the RobustScaler
        self.X_scaler = RobustScaler(quantile_range=(1, 99))
        X_scaled = self.X_scaler.fit_transform(X_gpu.drop('__id__', axis=1))
        self.feature_cols = list(X.columns)  # Store deduplicated features

        # Initialize and fit the target scaler
        self.y_scaler = RobustScaler()
        y_scaled = self.y_scaler.fit_transform(y.values.reshape(-1, 1))

        print("\nBefore scaling:")
        print(f"Y range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"X range: [{X.min().min():.2f}, {X.max().max():.2f}]")
        
        # Convert to GPU tensors once
        X_gpu = cudf.DataFrame.from_pandas(X)
        y_gpu = cudf.Series(y)
        
        # Initialize scalers ONCE with consistent parameters
        self.X_scaler = RobustScaler(quantile_range=(0.1, 99.9))
        self.y_scaler = RobustScaler(quantile_range=(0.1, 99.9))
        
        # Fit and transform
        X_scaled = self.X_scaler.fit_transform(X_gpu)
        y_scaled = self.y_scaler.fit_transform(y_gpu.values.reshape(-1, 1))
        
        # Minimal scaling verification
        print("\nScaling ranges:")
        print(f"X: [{X.min().min():.2f}-{X.max().max():.2f}] -> [{float(X_scaled.min().min()):.2f}-{float(X_scaled.max().max()):.2f}]")
        print(f"y: [{y.min():.2f}-{y.max():.2f}] -> [{float(y_scaled.min()):.2f}-{float(y_scaled.max()):.2f}]")
        
        # Convert cuDF Series to Python lists properly, with numpy handling
        scaling_info = {
            'x_quantile_range': [float(x) for x in list(self.X_scaler.quantile_range)],
            'y_quantile_range': [float(x) for x in list(self.y_scaler.quantile_range)],
            'x_scale': [float(x) for x in (self.X_scaler.scale_.get() if hasattr(self.X_scaler.scale_, 'get') 
                                           else self.X_scaler.scale_.to_arrow().to_pylist())],
            'y_scale': [float(x) for x in (self.y_scaler.scale_.get() if hasattr(self.y_scaler.scale_, 'get') 
                                           else self.y_scaler.scale_.to_arrow().to_pylist())],
            'x_center': [float(x) for x in (self.X_scaler.center_.get() if hasattr(self.X_scaler.center_, 'get') 
                                            else self.X_scaler.center_.to_arrow().to_pylist())],
            'y_center': [float(x) for x in (self.y_scaler.center_.get() if hasattr(self.y_scaler.center_, 'get') 
                                            else self.y_scaler.center_.to_arrow().to_pylist())]
        }

        # Save scaling info to JSON
        scaling_path = self.output_dir / self.name / "model" / "scaling_info.json"
        with open(scaling_path, 'w') as f:
            json.dump(scaling_info, f, indent=2)
        
        # Track both training and validation scores
        best_train_score = float('-inf')
        best_val_score = float('-inf')
        best_model = None
        best_params = None
        
        # Add k-fold cross validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Track parameter trends
        param_scores = {
            'C': defaultdict(list),
            'epsilon': defaultdict(list)
        }

        for params in itertools.product(*param_grid.values()):
            C, epsilon, kernel = params
            current_iteration += 1
            
            # Track cross validation scores
            cv_train_scores = []
            cv_val_scores = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled.to_numpy()[train_idx], X_scaled.to_numpy()[val_idx]
                y_train, y_val = cudf.Series(y_scaled[train_idx].flatten()), cudf.Series(y_scaled[val_idx].flatten())
                
                model = self.create_model(C=C, epsilon=epsilon, kernel=kernel)
                model.fit(X_train, y_train)
                
                # Get both training and validation scores
                train_score = self.evaluate_model(model, X_train, y_train)
                val_score = self.evaluate_model(model, X_val, y_val)
                
                cv_train_scores.append(train_score)
                cv_val_scores.append(val_score)
            
            # Average scores across folds
            mean_train_score = np.mean(cv_train_scores)
            mean_val_score = np.mean(cv_val_scores)
            std_val_score = np.std(cv_val_scores)
            
            self.logger.info(f"Iteration {current_iteration}/{total_iterations}:")
            self.logger.info(f"Params: C={C}, epsilon={epsilon}, kernel={kernel}")
            self.logger.info(f"Train score: {mean_train_score:.4f}")
            self.logger.info(f"Val score: {mean_val_score:.4f} (±{std_val_score:.4f})")
            
            if mean_val_score > best_val_score:  # Change comparison to >
                best_val_score = mean_val_score
                best_train_score = mean_train_score
                best_model = model
                best_params = {"C": C, "epsilon": epsilon, "kernel": kernel}
                
                # Save predictions for all splits as before
                splits = {
                    'finaltrain': data_manager.finaltrain,
                    'finalvalidate': data_manager.finalvalidate,
                    'finaltest': data_manager.finaltest
                }
                for split_name, split_data in splits.items():
                    predictions = self.add_predictions(best_model, split_data, split_name, feature_cols)
                    output_path = self.output_dir / self.name / f"SVR_{split_name}_estimates_{best_val_score:.4f}.csv"
                    predictions.to_csv(output_path, index=False)
                    self.logger.info(f"Saved {split_name} predictions with score {best_val_score:.4f} to {output_path}")

            # Track scores for parameter analysis
            param_scores['C'][C].append((mean_train_score, mean_val_score, std_val_score))
            param_scores['epsilon'][epsilon].append((mean_train_score, mean_val_score, std_val_score))
            
            # Log trending information
            self.logger.info(f"\nIteration {current_iteration}/{total_iterations}:")
            self.logger.info(f"Params: C={C}, epsilon={epsilon}")
            self.logger.info(f"Train score: {mean_train_score:.4f}")
            self.logger.info(f"Val score: {mean_val_score:.4f} (±{std_val_score:.4f})")
            self.logger.info(f"Score gap: {mean_train_score - mean_val_score:.4f}")

        totaltime = time.time() - start_time
        self.logger.info(f"Experiment completed in {totaltime:.2f} seconds.")
        
        # Save the best model and metadata into the model folder
        self.save_model(best_model, best_params, best_val_score)
        
        # Generate a final submission file from the final test set and save it in output/Svr/model
        submission = self.add_predictions(best_model, data_manager.finaltest.copy(), "finaltest", feature_cols)
        submission_path = self.output_dir / self.name / "model" / "submission.csv"
        submission.to_csv(submission_path, index=False)
        self.logger.info(f"Saved final submission file to {submission_path}")
        
        return best_model, {
            "best_train_score": best_train_score, 
            "best_val_score": best_val_score,
            "param_scores": param_scores
        }

    def add_predictions(self, model, data, split_name, feature_cols):
        X = self.dropIdPremium(data)

        # Ensure the feature order matches the one used during scaling and no duplicates
        X = X[self.feature_cols]  # Use stored deduplicated features

        # Scale the features using the same scaler used during training
        X_gpu = cudf.DataFrame.from_pandas(X)
        X_scaled = self.X_scaler.transform(X_gpu)  # Use transform instead of to_numpy
        
        predictions_scaled = model.predict(X_scaled)
        if isinstance(predictions_scaled, cudf.Series):
            predictions_scaled = predictions_scaled.values
        predictions_scaled = predictions_scaled.reshape(-1, 1)

        # Inverse transform the predictions
        predictions = self.y_scaler.inverse_transform(predictions_scaled).flatten()
        
        # Verify scaling at each step
        print(f"\nPrediction Scaling Verification ({split_name}):")
        print(f"1. Input features:")
        print(f"   Original: [{X.min().min():.2f}, {X.max().max():.2f}]")
        print(f"   Scaled:   [{float(X_scaled.min().min()):.2f}, {float(X_scaled.max().max()):.2f}]")
        print(f"2. Predictions:")
        print(f"   Scaled:   [{float(predictions_scaled.min()):.2f}, {float(predictions_scaled.max()):.2f}]")
        print(f"   Final:    [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            self.logger.info(f"{split_name} RMSLE: {score:.3f}")
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': np.round(predictions.get(), 3)
        })
        
        return submission_df

    def log_parameter_performance(self, param_scores):
        """Log the performance of each parameter value."""
        for param_name, scores in param_scores.items():
            self.logger.info(f"\n{param_name} Performance Summary:")
            for value, (train_scores, val_scores, stds) in sorted(scores.items()):
                mean_train = np.mean(train_scores)
                mean_val = np.mean(val_scores)
                mean_std = np.mean(stds)
                self.logger.info(
                    f"{param_name}={value:.6f}: "
                    f"Train={mean_train:.4f}, "
                    f"Val={mean_val:.4f} (±{mean_std:.4f})"
                )

if __name__ == "__main__":
    output_dir = Path('output')
    experiment = SVRExperiment(output_dir)
    
    try:
        experiment.logger.info("Starting new SVR experiment...")
        best_model, results = experiment.run()
        experiment.logger.info("\nResults:")
        experiment.logger.info(results)
    except Exception as e:
        experiment.logger.error(f"An error occurred: {e}", exc_info=True)