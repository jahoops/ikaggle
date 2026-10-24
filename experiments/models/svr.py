import logging
import time
from pathlib import Path
import sys
import itertools
import json
from datetime import datetime
import platform
import joblib

# Ensure experiments module is in1 path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from cuml.svm import SVR, LinearSVR
from cuml.preprocessing import RobustScaler
import cudf
import cupy as cp
import numpy as np
import pandas as pd
from experiments.data_manager import data_manager
from experiments.base_experiment import BaseExperiment




logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def safe_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error (RMSLE) safely."""
    # Convert to numpy if needed
    if hasattr(y_true, 'values'):
        y_true = y_true.values
    if hasattr(y_pred, 'values'):
        y_pred = y_pred.values
    
    assert len(y_true) == len(y_pred)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return np.sqrt(np.mean((log_true - log_pred) ** 2))

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
        
        # Configure logging
        log_file = self.run_dir / f"{self.name}_{self.run_id}.log"
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        
        # Terminal handler with less detailed formatting
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(message)s')
        )
        
        # Configure logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        # Log system info
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
        # Verify GPU availability
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
        """Save model and metadata to disk."""
        model_dir = self.output_dir / self.name / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using joblib
        model_path = model_dir / "best_model.joblib"
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "params": params,
            "score": float(score),  # Convert to Python float for JSON
            "timestamp": self.format_time(time.time())
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved model to {model_path}")

    def load_model(self):
        """Load saved model and metadata."""
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
        predictions = model.predict(X)
        score = safe_rmsle(y, predictions)
        return score

    def run(self):
        start_time = time.time()
        run_id = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / self.name / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        best_score = float('inf')
        best_model = None
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            'epsilon': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            'kernel': ['linear']
        }
        total_iterations = len(list(itertools.product(*param_grid.values())))
        current_iteration = 0

        data = data_manager.finaltrain.copy()
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']

        X_gpu = cudf.DataFrame.from_pandas(X)
        y_gpu = cudf.Series(y)

        self.X_scaler = RobustScaler()
        self.y_scaler = RobustScaler()
        X_scaled = self.X_scaler.fit_transform(X_gpu)
        y_scaled = self.y_scaler.fit_transform(y_gpu.values.reshape(-1, 1)).flatten()

        for params in itertools.product(*param_grid.values()):
            current_iteration += 1
            C, epsilon, kernel = params
            
            print(f"Training with C={C}, epsilon={epsilon}, kernel={kernel}")
            
            model = self.create_model(C=C, epsilon=epsilon, kernel=kernel)
            model.fit(X_scaled, y_scaled)
            score = self.evaluate_model(model, X_scaled, y_scaled)
            
            print(f"Iteration {current_iteration}/{total_iterations} - Score: {score}")
            
            if score < best_score:
                best_score = score
                best_model = model
                
                # Save predictions for all splits
                splits = {
                    'finaltrain': data_manager.finaltrain,
                    'finalvalidate': data_manager.finalvalidate,
                    'finaltest': data_manager.finaltest
                }
                
                for split_name, split_data in splits.items():
                    predictions = self.add_predictions(best_model, split_data, split_name)
                    output_path = self.output_dir / self.name / f'SVR_{split_name}_estimates_{best_score:.3f}.csv'
                    predictions.to_csv(output_path, index=False)
        
        return best_model, {"best_score": best_score}

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        X_gpu = cudf.DataFrame.from_pandas(X)
        X_scaled = self.X_scaler.transform(X_gpu)
        
        predictions_scaled = model.predict(X_scaled)
        if isinstance(predictions_scaled, cudf.Series):
            predictions_scaled = predictions_scaled.values
        predictions_scaled = predictions_scaled.reshape(-1, 1)
        
        dollar_predictions = self.y_scaler.inverse_transform(predictions_scaled)
        if isinstance(dollar_predictions, cp.ndarray):
            dollar_predictions = dollar_predictions.get()
        dollar_predictions = dollar_predictions.flatten()
        
        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], dollar_predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"
        
        return pd.DataFrame({
            'id': data['id'],
            'Premium Amount': dollar_predictions.round(3),
            'Confidence': np.zeros_like(dollar_predictions).round(3)
        })

if __name__ == "__main__":
    output_dir = Path('output')
    experiment = SVRExperiment(output_dir)
    
    try:
        # Run new experiment
        experiment.logger.info("Starting new SVR experiment...")
        best_model, results = experiment.run()
        
        # Display results
        experiment.logger.info("\nResults:")
        
    except Exception as e:
        experiment.logger.error(f"An error occurred: {e}", exc_info=True)