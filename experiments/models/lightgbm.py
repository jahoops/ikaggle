import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from experiments.base_experiment import BaseExperiment
from experiments import data_manager, safe_rmsle

logger = logging.getLogger(__name__)

class LightGBMExperiment(BaseExperiment):
    def __init__(self, output_dir, test_mode=False):
        super().__init__(output_dir, test_mode)
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "LightGBM"

    def create_model(self):
        """Instantiate LightGBM model."""
        return lgb.LGBMRegressor(random_state=42)

    def get_param_distributions(self):
        """Define hyperparameter distributions for RandomizedSearchCV."""
        if self.test_mode:
            return {
                'n_estimators': [50],
                'num_leaves': [31],
                'max_depth': [-1],
                'learning_rate': [0.1]
            }
        else:
            return {
                'n_estimators': [100, 200],
                'num_leaves': [31, 63],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.05, 0.1]
            }

    def run(self):
        """Run the LightGBM experiment and return the trained model."""
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

            # Load final preprocessed data
            data = data_manager.finaltrain.copy()

            param_distributions = self.get_param_distributions()
            self.logger.info(f"Param distributions: {param_distributions}")

            cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
            search = RandomizedSearchCV(
                estimator=self.create_model(),
                param_distributions=param_distributions,
                n_iter=1 if self.test_mode else 10,
                scoring='neg_mean_squared_error',
                cv=cv,
                verbose=1
            )

            X = self.dropIdPremium(data)
            y = data['PremiumAmount']

            search.fit(X, y)
            trained_model = search.best_estimator_
            self.logger.info(f"Best params: {search.best_params_}")

            # Generate predictions for train/validate/test
            self.add_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
            self.add_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
            self.add_predictions(trained_model, data_manager.finaltest, 'finaltest')

            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
            return trained_model

        except Exception as e:
            self.logger.error(f"[{self.name}] run failed: {str(e)}")
            raise

    def add_predictions(self, model, data, split_name):
        """Generate predictions with confidence and save CSV."""
        X = self.dropIdPremium(data)
        predictions = model.predict(X)

        # Calculate per-tree predictions to estimate standard deviation
        n_trees = model.n_estimators_
        tree_preds = []
        for i in range(n_trees):
            # Predict up to tree i+1
            partial_pred = model.predict(X, num_iteration=i+1)
            tree_preds.append(partial_pred)
        tree_preds = np.array(tree_preds)  # shape (n_trees, n_samples)
        
        pred_std = np.std(tree_preds, axis=0)

        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': pred_std.round(3)
        })

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"

        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

def run_lightgbm(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = LightGBMExperiment(output_path, test_mode=test_mode)
    return experiment.run()

if __name__ == "__main__":
    run_lightgbm('output', test_mode=True)
