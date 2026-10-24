from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing  import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.utils import check_array
import os
import sys
import joblib
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from experiments.data_manager import data_manager
from utils import safe_rmsle, feature_cols
from experiments.base_experiment import BaseExperiment  # Updated import
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

class RandomForestExperiment(BaseExperiment):
    def __init__(self, output_dir, test_mode=False, sample_size=100000):
        super().__init__(output_dir, test_mode)  # Corrected super() call
        self.sample_size = sample_size

    @property
    def name(self) -> str:
        return "RandomForest"
    
    def create_model(self):
        def select_features(X):
            # Define your specific feature list
            return X[feature_cols]
        
        # Custom transformer to select specific features
        feature_selector = FunctionTransformer(select_features, validate=False)
        
        return Pipeline([
            ('preprocessor', feature_selector),
            ('random_forest', RandomForestRegressor(random_state=42, n_jobs=-1))  # Use all available cores
        ])

    print("feature_cols: ", feature_cols)
    def get_param_distributions(self):
        if self.test_mode:
            return {
                'random_forest__n_estimators': [50],
                'random_forest__max_depth': [None],
                'random_forest__min_samples_split': [2],
                'random_forest__min_samples_leaf': [1],
                'random_forest__max_features': ['sqrt'],
                'random_forest__max_samples': [0.8, 1.0]
            }
        else:
            return {
                'random_forest__n_estimators': [100, 200, 300],
                'random_forest__max_depth': [None, 10, 20, 30],
                'random_forest__min_samples_split': [2, 5, 10],
                'random_forest__min_samples_leaf': [1, 2, 4],
                'random_forest__max_features': ['sqrt', 'log2'],
                'random_forest__max_samples': [0.8, 0.9, 1.0]
            }

    def run(self):
        """Run the Random Forest experiment and return the trained model."""
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")
            
            # Load data
            data = data_manager.finaltrain.copy()
            X = data.drop(columns=['id', 'PremiumAmount'])
            y = data['PremiumAmount']

            # Ensure feature names are consistent during training and prediction
            X_train_checked = check_array(X, ensure_2d=True, dtype=None, estimator='RandomForestRegressor')
            print(f"X_train_checked: {X_train_checked}, feature_cols: {feature_cols} X: {X.columns}")
            X = X[feature_cols]
            print(f"X: {X.columns}")
            cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
            param_distributions = self.get_param_distributions()
            self.logger.info(f"Param distributions: {param_distributions}")

            rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

            # Create the pipeline model
            model = self.create_model()

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=1 if self.test_mode else 10,
                scoring=rmsle_scorer,
                cv=cv,
                verbose=5  # Increase verbosity to get detailed progress
            )
            search.fit(X, y)
            trained_model = search.best_estimator_
            self.logger.info(f"Best params: {search.best_params_}")

            # Add predictions for train/validate/test
            self.add_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
            self.add_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
            self.add_predictions(trained_model, data_manager.finaltest, 'finaltest')

            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
            return trained_model
                
        except Exception as e:
            self.logger.error(f"RandomForest run failed: {str(e)}")
            raise

    def add_predictions(self, model, data, split_name):
        try:
            X = self.dropIdPremium(data)
            # Ensure all features are numeric
            X = X.apply(pd.to_numeric, errors='coerce')

            # Ensure feature names are consistent
            X.columns = model.feature_names_in_

            predictions = model.predict(X)
            self.logger.info(f"Generated predictions for {split_name}")

            # Standard deviation across RandomForest estimators:
            rf = model.named_steps['random_forest']
            all_estimator_preds = [est.predict(X) for est in rf.estimators_]
            pred_std = np.std(all_estimator_preds, axis=0)

            if split_name in ['finaltrain', 'finalvalidate']:
                score = rmsle(data['PremiumAmount'], predictions)
                filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
                self.logger.info(f"{split_name} score: {score:.3f}")
            else:
                filename = f"{self.name}_{split_name}_estimates.csv"

            submission = pd.DataFrame({
                'id': data['id'],
                'Premium Amount': predictions.round(3),
                'Confidence': pred_std.round(3)
            })

            output_path = self.output_dir / filename
            submission.to_csv(output_path, index=False)
            self.logger.info(f"Saved {split_name} predictions to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to add predictions for {split_name}: {str(e)}")
            raise

def run_randomforest(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = RandomForestExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_randomforest('output/RandomForest', test_mode=True)
    else:
        run_randomforest('output/RandomForest', test_mode=False)