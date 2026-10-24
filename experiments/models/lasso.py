import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from experiments.base_experiment import BaseExperiment
from experiments import data_manager, safe_rmsle

logger = logging.getLogger(__name__)

class LassoExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "Lasso"

    def create_model(self):
        return Pipeline([
            ('lasso', Lasso(max_iter=10000))
        ])

    def get_param_distributions(self):
        if self.test_mode:
            return {
                'lasso__alpha': [1.0],
                'lasso__fit_intercept': [True]
            }
        else:
            return {
                'lasso__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'lasso__fit_intercept': [True, False]
            }

    def run(self):
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

            data = data_manager.finaltrain.copy()
            X = self.dropIdPremium(data)
            y = data['PremiumAmount']

            param_grid = self.get_param_distributions()
            self.logger.info(f"Param distributions: {param_grid}")

            grid_search = GridSearchCV(
                self.create_model(),
                param_grid,
                cv=3,
                scoring=self.get_scoring_metric(),
                verbose=1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            self.logger.info(f"Best params: {grid_search.best_params_}")

            if self.test_mode:
                self.logger.info("Test mode: True")

            self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
            self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate') 
            self.add_predictions(best_model, data_manager.finaltest, 'finaltest')

            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
            return best_model

        except Exception as e:
            self.logger.error(f"[{self.name}] run failed: {str(e)}")
            raise

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)
        
        # Zero confidence for non-ensemble model
        confidence = np.zeros_like(predictions)

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"

        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': confidence.round(3)
        })

        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

    def get_scoring_metric(self):
        return make_scorer(safe_rmsle, greater_is_better=False)