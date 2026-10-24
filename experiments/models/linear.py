import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from experiments.base_experiment import BaseExperiment
from experiments import data_manager, safe_rmsle

logger = logging.getLogger(__name__)

class FeatureWeightTransformer:
    def __init__(self):
        self.feature_weights = {
            'HealthScore': 0.5523, 'AnnualIncome': 0.2245,
            'CreditScore': 0.0800, 'Age': 0.0085
        }
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_weighted = X.copy()
        for feat, weight in self.feature_weights.items():
            if feat in X_weighted.columns:
                X_weighted[feat] = X_weighted[feat] * weight
        return X_weighted

class LinearRegressionExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "LinearRegression"

    def create_model(self):
        return Pipeline([
            ('feature_weights', FeatureWeightTransformer()),
            ('linear', LinearRegression())
        ])

    def get_param_distributions(self):
        return {
            'linear__fit_intercept': [True, False],
            'linear__copy_X': [True, False],
            'linear__positive': [False]  # or [True, False] if desired
        }

    def run(self):
        start_time = time.time()
        self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

        data = data_manager.finaltrain.copy()
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']

        param_grid = self.get_param_distributions()
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

        # Save predictions
        self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
        self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate')
        self.add_predictions(best_model, data_manager.finaltest, 'finaltest')

        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
        return best_model

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)

        # Confidence = zeros for non-ensemble linear models
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

class RidgeRegressionExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "RidgeRegression"

    def create_model(self):
        return Pipeline([
            ('feature_weights', FeatureWeightTransformer()),
            ('ridge', Ridge(max_iter=10000))
        ])

    def get_param_distributions(self):
        return {
            'ridge__alpha': [0.01, 0.1, 1.0],
            'ridge__fit_intercept': [True, False],
            'ridge__positive': [False, True]
        }

    def run(self):
        start_time = time.time()
        logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

        data = data_manager.finaltrain.copy()
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']

        param_grid = self.get_param_distributions()
        grid_search = GridSearchCV(
            self.create_model(),
            param_grid,
            cv=3,
            scoring=self.get_scoring_metric(),
            verbose=1
        )
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        best_score = -grid_search.best_score_  # Because greater_is_better=False

        self.logger.info(f"Best params: {grid_search.best_params_}")

        # Add predictions
        self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
        self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate')
        self.add_predictions(best_model, data_manager.finaltest, 'finaltest')

        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
        return best_model

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)

        # Confidence = zeros
        confidence = np.zeros_like(predictions)

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"

        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': confidence.round(3)
        })

        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} predictions to {output_path}")

    def get_scoring_metric(self):
        return make_scorer(safe_rmsle, greater_is_better=False)

class XGBoostExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "XGBoost"

    def create_model(self):
        return Pipeline([
            ('scaler', RobustScaler()),
            ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])

    def get_param_distributions(self):
        if self.test_mode:
            return {
                'xgb__n_estimators': [100],
                'xgb__learning_rate': [0.1],
                'xgb__max_depth': [3],
                'xgb__subsample': [0.8],
                'xgb__colsample_bytree': [0.8]
            }
        else:
            return {
                'xgb__n_estimators': [50, 100, 200],
                'xgb__learning_rate': [0.01, 0.1, 0.2],
                'xgb__max_depth': [3, 5, 7],
                'xgb__subsample': [0.6, 0.8, 1.0],
                'xgb__colsample_bytree': [0.6, 0.8, 1.0]
            }

    def run(self):
        try:
            start_time = time.time()
            self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")

            data = data_manager.finaltrain.copy()
            X = self.dropIdPremium(data)
            y = data['PremiumAmount']

            cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
            param_distributions = self.get_param_distributions()

            search = RandomizedSearchCV(
                estimator=self.create_model(),
                param_distributions=param_distributions,
                n_iter=1 if self.test_mode else 10,
                scoring='neg_mean_squared_error',
                cv=cv,
                verbose=1
            )
            search.fit(X, y)
            trained_model = search.best_estimator_
            self.logger.info(f"Best params: {search.best_params_}")

            self.add_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
            self.add_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
            self.add_predictions(trained_model, data_manager.finaltest, 'finaltest')

            end_time = time.time()
            elapsed = end_time - start_time
            self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
            return trained_model

        except Exception as e:
            self.logger.error(f"XGBoost run failed: {str(e)}")
            raise

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        predictions = model.predict(X)

        # Estimate confidence by collecting partial predictions
        xgb_est = model.named_steps['xgb']
        n_estimators = xgb_est.get_params()['n_estimators']
        partials = []
        for i in range(n_estimators):
            # Predict up to tree i+1
            part_pred = xgb_est.predict(X, iteration_range=(0, i+1))
            partials.append(part_pred)
        partials = np.array(partials)
        pred_std = np.std(partials, axis=0)

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
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

def run_linear(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = LinearRegressionExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

def run_ridge(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = RidgeRegressionExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    run_linear('output', test_mode=True)
    run_ridge('output', test_mode=True)
