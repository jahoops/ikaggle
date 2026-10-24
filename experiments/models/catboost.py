# filepath: /mnt/c/home/USER/ensemble/experiments/models/catboost.py
import logging
import time
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from catboost import CatBoostRegressor
from experiments.base_experiment import BaseExperiment
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import data_manager, safe_rmsle

logger = logging.getLogger(__name__)

class CatBoostExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "CatBoost"
    
    def __init__(self, output_dir, test_mode=False, sample_size=100000):
        super().__init__(output_dir, test_mode)
        self.sample_size = sample_size
    
    def create_model(self):
        """Instantiate the CatBoost model."""
        return CatBoostRegressor(random_state=42, verbose=0)

    def get_param_distributions(self):
        """Define hyperparameter distributions for RandomizedSearchCV."""
        if self.test_mode:
            return {
                'depth': [4],
                'learning_rate': [0.1],
                'iterations': [100],
                'l2_leaf_reg': [1]
            }
        else:
            return {
                'depth': [4, 6],
                'learning_rate': [0.01, 0.05],
                'iterations': [250, 500],
                'l2_leaf_reg': [1, 3, 5]
            }

    def run(self):
        start_time = time.time()
        self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")
        
        # Load final preprocessed data
        data = data_manager.finaltrain.copy()
        
        # Debug: verify PremiumAmount positivity
        neg_count = (data["PremiumAmount"] <= 0).sum()
        min_premium = data["PremiumAmount"].min()
        max_premium = data["PremiumAmount"].max()
        self.logger.info(f"PremiumAmount negatives/zeros: {neg_count}, min={min_premium}, max={max_premium}")

        # Prepare param distributions and cross-validation
        param_distributions = self.get_param_distributions()
        cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)

        search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=cv,
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']
        
        search.fit(X, y, early_stopping_rounds=50, eval_set=(X, y))
        trained_model = search.best_estimator_
        
        model_path = self.output_dir / f"{self.name}_model.cbm"
        trained_model.save_model(str(model_path))
        self.logger.info(f"[{self.name}] Saved CatBoost model to {model_path}")

        self.add_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
        self.add_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
        self.add_predictions(trained_model, data_manager.finaltest, 'finaltest')

        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)
        
        # Get confidence intervals from trees
        trees_predictions = np.array([
            model.predict(X, ntree_start=i, ntree_end=i+1) 
            for i in range(model.tree_count_)
        ])
        pred_std = np.std(trees_predictions, axis=0)
        
        # Scale confidence to 0-100 range like KNN
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
        # Save predictions
        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

def run_catboost(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = CatBoostExperiment(output_path, test_mode=test_mode)
    experiment.run()

if __name__ == "__main__":
    run_catboost('output', test_mode=True)