import sys
import logging
import cudf
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import xgboost as xgb
import time
from cuml.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from experiments.data_manager import data_manager  # Fix import
from sklearn.metrics import mean_squared_log_error
from experiments.base_experiment import BaseExperiment
from utils import safe_rmsle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class XGBoostExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "XGBoost"
    
    def __init__(self, output_dir, test_mode=False):
        super().__init__(output_dir, test_mode)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple logging setup
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Initialize scalers
        self.X_scaler = RobustScaler()
        self.y_scaler = RobustScaler()

        self.feature_weights = {
            'HealthScore': 0.5523, 'AnnualIncome': 0.2245,
            'CreditScore': 0.0800, 'Age': 0.0085,
            'SignupYear': 0.0023, 'PreviousClaims': 0.0021,
            'NumberofDependents': 0.0018, 'Location_Rural': 0.0015,
            'VehicleAge': 0.0015, 'InsuranceDuration': 0.0015
        }

    def transform(self, X):
        X_weighted = X.copy()
        for feat, weight in self.feature_weights.items():
            if feat in X.columns:
                X_weighted[feat] = X[feat] * np.sqrt(weight)
        return X_weighted
    
    
    def create_model(self):
        """Create and return XGBoost model instance"""
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            device='CUDA'
        )

    def run(self):
        start_time = time.time()
        self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")
        
        data = data_manager.finaltrain.copy()
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']

        dtrain = xgb.DMatrix(X, label=y)
        
        base_params = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'device': 'cuda',
            'random_state': 42,
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.05,
        }
        
        best_score = float('inf')
        best_model = None
        best_iteration = 0
        
        for n_round in range(1000, 10001, 1000):
            model = xgb.train(
                params=base_params,
                dtrain=dtrain,
                num_boost_round=n_round
            )
            
            # Calculate RMSLE score
            train_preds = model.predict(dtrain)
            score = safe_rmsle(y, train_preds)
            
            self.logger.info(f"Iteration {n_round}, RMSLE: {score:.3f}")
            
            if score < best_score:
                best_score = score
                best_model = model
                best_iteration = n_round
                
                # Save predictions only when we have a new best model
                self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
                self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate')
                self.add_predictions(best_model, data_manager.finaltest, 'finaltest')
                
                self.logger.info(f"New best model at iteration {n_round}, RMSLE: {score:.3f}")
        
        end_time = time.time()
        self.logger.info(f"Training completed in {end_time - start_time:.2f}s")
        self.logger.info(f"Best RMSLE: {best_score:.3f} at iteration {best_iteration}")
        
        return best_model, {"best_score": best_score}

    def safe_rmsle(y_true, y_pred):
        """Calculate Root Mean Squared Logarithmic Error (RMSLE) safely."""
        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        dmatrix = xgb.DMatrix(X)
        predictions = model.predict(dmatrix)
        
        # Calculate confidence scores using tree variance
        trees_predictions = np.array([
            model.predict(dmatrix, iteration_range=(i, i+1))
            for i in range(model.num_boosted_rounds())
        ])
        pred_std = np.std(trees_predictions, axis=0)
        scaled_conf = 100 * (pred_std - pred_std.min()) / (pred_std.max() - pred_std.min())
        
        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': scaled_conf.round(3)
        })
        
        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"
        
        output_path = self.run_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

if __name__ == "__main__":
    output_dir = Path('output')
    runthis = XGBoostExperiment(output_dir)
    
    try:
        model, _ = runthis.run()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
