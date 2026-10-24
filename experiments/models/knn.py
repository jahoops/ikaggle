# filepath: /home/USER/Kaggle/InsuranceNext/experiments/models/knn.py

import logging
import time
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from experiments.base_experiment import BaseExperiment
from pathlib import Path
import pandas as pd
import numpy as np
from utils import safe_rmsle

logger = logging.getLogger(__name__)

class KNNExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "KNN"

    def __init__(self, output_dir, test_mode=False, sample_size=100000):
        super().__init__(output_dir, test_mode)
        self.sample_size = sample_size

    def create_model(self):
        """Instantiate the KNN model in a pipeline."""
        return Pipeline([
            ('knn', KNeighborsRegressor())
        ])

    def get_param_distributions(self):
        """Define hyperparameter distributions for RandomizedSearchCV."""
        if self.test_mode:
            return {
                'knn__n_neighbors': [5],
                'knn__weights': ['uniform'],
                'knn__p': [2]
            }
        else:
            return {
                'knn__n_neighbors': [3, 5, 7, 9],
                'knn__weights': ['uniform', 'distance'],
                'knn__p': [1, 2]
            }

    def run(self):
        start_time = time.time()
        self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

        # Load base data
        data = data_manager.finaltrain.copy()

        # Prepare data
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']

        # Prepare cross-validation
        cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
        param_distributions = self.get_param_distributions()
        
        search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=20,
            scoring='neg_mean_squared_error',
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=2
        )
        search.fit(X, y)
        best_model = search.best_estimator_
        
        self.logger.info(f"Best params: {search.best_params_}")

        # Save predictions using add_predictions
        self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
        self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate')
        self.add_predictions(best_model, data_manager.finaltest, 'finaltest')

        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
        return best_model

    def add_predictions(self, model, data, split_name):
        """Save predictions with confidence and scoring"""
        X = self.dropIdPremium(data)
        predictions = model.predict(X)
        confidence = self.calculate_confidence(model, X)  # Called from base class
        
        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3), 
            'Confidence': confidence.round(3)  # Included in output
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

    def calculate_confidence(self, model, X):
        """Calculate confidence scores for a KNN pipeline."""
        # Extract the KNN step
        knn_estimator = model.named_steps['knn']
        
        distances, _ = knn_estimator.kneighbors(X)
        mean_dist = np.mean(distances, axis=1)
        std_dist = np.std(distances, axis=1)
        
        raw_conf = 1 / (mean_dist + std_dist + 1e-6)
        scaled_conf = 100 * (raw_conf - raw_conf.min()) / (raw_conf.max() - raw_conf.min())
        
        return scaled_conf

def run_knn(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = KNNExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    run_knn('output', test_mode=True)