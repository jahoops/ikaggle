import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, train_test_split  # Add this import
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from .base_experiment import BaseExperiment, feature_importances  # Add this import
from .utils import safe_rmsle, save_model, FeatureNamePreserver

logger = logging.getLogger(__name__)

class XGBRegressorWrapper(XGBRegressor, BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _more_tags(self):
        return {
            'non_deterministic': True,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multioutput_only': False,
            'binary_only': False,
            'requires_fit': True,
            'preserves_dtype': [np.float64],
            'requires_y': True,
            'pairwise': False,
        }

class XGBoostExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "XGBoost"
        
    def create_model(self):
        return Pipeline([
            ('names', FeatureNamePreserver()),
            ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
    
    def run(self):
        """Override run to implement chunked training"""
        try:
            X, y = self.load_data()
            self.logger.info(f"Loaded data: X shape={X.shape}, y shape={y.shape}")
            
            # Ensure target values are non-negative
            if (y < 0).any():
                raise ValueError("Target values contain negative values, which are not allowed for mean_squared_log_error.")
            
            # More aggressive sampling for XGBoost
            if len(X) > 10000:
                self.logger.info("Large dataset detected, using stratified sampling")
                X_sample, _, y_sample, _ = train_test_split(
                    X, y, 
                    train_size=5000,     # Reduced sample size
                    random_state=42,
                    stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
                )
                X, y = X_sample, y_sample
            
            self.logger.info(f"Training XGBoost with {len(X)} samples")
            
            # Create pipeline with proper structure
            pipeline = Pipeline([
                ('names', FeatureNamePreserver()),
                ('scaler', RobustScaler()),
                ('xgb', self.create_model())
            ])
            
            # Define correct XGBoost parameter distributions
            param_distributions = {
                'xgb__n_estimators': [100, 200, 300, 500],
                'xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
                'xgb__max_depth': [3, 6, 9, 12],
                'xgb__min_child_weight': [1, 3, 5, 7],
                'xgb__subsample': [0.6, 0.8, 1.0],
                'xgb__colsample_bytree': [0.6, 0.8, 1.0],
                'xgb__gamma': [0, 0.1, 0.2]
            }
            
            # Use Repeated K-Fold
            rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=42)
            
            # Randomized search over parameter space
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_distributions,
                n_iter=10, scoring='neg_mean_squared_log_error',
                cv=rkf, verbose=2, random_state=42, n_jobs=-1
            )
            search.fit(X, y)
            
            # Best model
            best_pipeline = search.best_estimator_
            best_score_msle = -search.best_score_
            msle = best_score_msle if best_score_msle > 0 else 0.0
            final_rmsle = np.sqrt(msle)
            self.logger.info(f"Best XGBoost RMSLE: {final_rmsle:.4f}")
            
            # Best model info with additional XGBoost-specific information
            additional_info = {
                'best_params': search.best_params_,
                'cv_results': {
                    'mean_test_score': float(search.cv_results_['mean_test_score'][search.best_index_]),
                    'std_test_score': float(search.cv_results_['std_test_score'][search.best_index_])
                },
                'feature_importance': dict(zip(
                    X.columns,
                    best_pipeline.named_steps['xgb'].feature_importances_
                ))
            }
            
            model_info = self.create_model_info(
                model=best_pipeline,
                X=X,
                y=y,
                score=final_rmsle,
                additional_info=additional_info
            )
            
            self.logger.info(f"Saving XGBoost model with score: {final_rmsle:.4f}")
            save_model(model_info, self.name, self.output_dir)
            
            self.logger.info("XGBoost model saved successfully")
            return model_info
            
        except Exception as e:
            error_msg = f"{self.name} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'model': None,
                'features': [],
                'score': float('inf'),
                'status': 'failed',
                'error': error_msg
            }

def run_xgboost(output_dir):
    experiment = XGBoostExperiment(output_dir)
    model_info = experiment.run()
    
    # Load existing state
    state = load_state(output_dir)
    
    if model_info['status'] == 'success':
        # Append as a dict with 'name' and 'score'
        state['completed'].append({
            'name': model_info['name'],
            'score': model_info['score']
        })
    elif model_info['status'] == 'failed':
        state['failed'].append((model_info['name'], model_info['error']))
    else:
        state['crashed'].append(model_info['name'])
    
    # Save updated state
    save_state(output_dir, state['completed'], state['failed'], state['crashed'])
    
    return model_info

if __name__ == "__main__":
    run_xgboost('output')