import multiprocessing
import psutil
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LassoCV
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, RegressorMixin
import optuna.integration.lightgbm as lgb_optuna
from scipy.stats import rankdata
from sklearn.metrics import make_scorer
from tqdm.auto import tqdm
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Custom weighted ensemble with online weight adjustment"""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else np.ones(len(models)) / len(models)
    
    def fit(self, X, y):
        self.meta_predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        # Adjust weights based on individual RMSLE scores
        scores = [-rmsle(y, pred) for pred in self.meta_predictions.T]
        self.weights = softmax(scores)
        return self
    
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return np.average(predictions, weights=self.weights, axis=1)

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    y_pred = np.clip(y_pred, 0, None)  # Clip negative predictions to 0
    return np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2)))

def analyze_feature_importance(model, feature_names):
    """Analyze and print feature importance"""
    try:
        # Get importances based on model type
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
            importance_dict = model.get_booster().get_score(importance_type='gain')
            # Match feature names with importance scores
            importances = np.zeros(len(feature_names))
            for fname, imp_value in importance_dict.items():
                try:
                    # Handle f0, f1, etc. feature names from XGBoost
                    if fname.startswith('f'):
                        idx = int(fname[1:])
                        if idx < len(feature_names):
                            importances[idx] = imp_value
                    # Handle actual feature names
                    elif fname in feature_names:
                        idx = list(feature_names).index(fname)
                        importances[idx] = imp_value
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not process feature {fname}: {str(e)}")
                    continue
        else:
            print("Warning: Model does not provide feature importances")
            return pd.DataFrame()
        
        # Create DataFrame with matched lengths
        importance_df = pd.DataFrame({
            'feature': np.array(feature_names)[:len(importances)],
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(importance_df.head(20))
        return importance_df
        
    except Exception as e:
        print(f"Error in analyze_feature_importance: {str(e)}")
        print(f"Feature names length: {len(feature_names)}")
        print(f"Importances length: {len(importances) if 'importances' in locals() else 'N/A'}")
        return pd.DataFrame()  # Return empty DataFrame on error

def optimize_xgb_params(X_train, y_train, n_trials=50):
    """Optimize XGBoost parameters using Optuna with explicit GPU settings"""
    def objective(trial):
        xgb_params = {
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'tree_method': 'hist',
            'device': 'cuda',  # Only specify device, remove gpu_id
            'predictor': 'gpu_predictor',
            'n_jobs': get_optimal_workers(),
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            model = xgb.XGBRegressor(**xgb_params, objective='reg:squarederror')
            model.fit(X_fold_train, y_fold_train)
            pred = model.predict(X_fold_val)
            score = -rmsle(y_fold_val, pred)
            scores.append(score)
            
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def create_meta_features(X, y, model):
    """Create meta-features from base model predictions"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        meta_features[val_idx] = model.predict(X[val_idx])
    
    return meta_features.reshape(-1, 1)

def create_advanced_meta_features(X, y, models):
    """Create advanced meta-features using multiple base models"""
    meta_features = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Basic predictions
    for model in models:
        meta_features.append(create_meta_features(X, y, model))
    
    # Rank features
    meta_features.append(rankdata(meta_features[0]).reshape(-1, 1))
    
    # Weighted average
    meta_features.append(np.average(meta_features[:3], axis=0, 
                                  weights=[0.4, 0.3, 0.3]))
    
    return np.hstack(meta_features)

def select_features(X_train, X_test, y_train, threshold='median'):
    """Select most important features using multiple models"""
    selectors = [
        ('xgb', xgb.XGBRegressor(n_estimators=100)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100)),
        ('gbm', GradientBoostingRegressor(n_estimators=100))
    ]
    
    selected_features = []
    for name, model in selectors:
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features.append(selector.get_support())
    
    # Take union of selected features
    final_mask = np.any(selected_features, axis=0)
    return X_train[:, final_mask], X_test[:, final_mask], final_mask

def lr_schedule(current_iter):
    """Dynamic learning rate schedule"""
    if current_iter < 100:
        return 0.01
    elif current_iter < 500:
        return 0.005
    else:
        return 0.001

def train_insurance_model(X_train, X_test, y_train, config, feature_cols):
    """Enhanced training with parallel processing"""
    use_gpu = validate_gpu()
    n_workers = get_optimal_workers()
    
    # Removed problematic backend n_jobs assignments
    # old_n_jobs = joblib.parallel.get_active_backend()[0].n_jobs
    # joblib.parallel.get_active_backend()[0].n_jobs = n_workers
    
    print("\n=== Starting Model Training ===")
    start_time = time.time()
    
    # Use parallel processing for feature selection
    print("\n1. Feature Selection (Parallel)")
    with Parallel(n_jobs=get_optimal_workers()) as parallel:
        selected_features = parallel(
            delayed(SelectFromModel(model, threshold='median').fit)(X_train, y_train)
            for _, model in [
                ('xgb', xgb.XGBRegressor(n_estimators=100, tree_method='hist', device='cuda')),
                ('lgb', lgb.LGBMRegressor(n_estimators=100, device='gpu')),
                ('gbm', GradientBoostingRegressor(n_estimators=100))
            ]
        )
    
    print("\n=== Starting Model Training ===")
    start_time = time.time()
    
    # Initialize progress bar for overall training
    pbar = tqdm(total=6, desc="Training Progress")
    
    print("\n1. Feature Selection")
    X_train_selected, X_test_selected, feature_mask = select_features(X_train, X_test, y_train)
    print(f"Selected {X_train_selected.shape[1]} out of {X_train.shape[1]} features")
    pbar.update(1)
    
    print("\n2. Creating Polynomial Features")
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_selected)
    X_test_poly = poly.transform(X_test_selected)
    print(f"Expanded to {X_train_poly.shape[1]} polynomial features")
    pbar.update(1)
    
    print("\n3. Parameter Optimization")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_params = optimize_xgb_params(X_train_selected, y_train, n_trials=30)
    print("Best parameters found:", best_params)
    pbar.update(1)
    
    print("\n4. Training Base Models")
    models = {}
    for name, model_func in [
        ("XGBoost", train_xgboost),
        ("LightGBM", train_lightgbm),
        ("GradientBoosting", lambda X, y: GradientBoostingRegressor(**config['gbm_params']).fit(X, y))
    ]:
        print(f"\nTraining {name}...")
        start = time.time()
        models[name] = model_func(X_train_selected, y_train)
        duration = time.time() - start
        
        # Calculate metrics
        train_pred = models[name].predict(X_train_selected)
        metrics = {
            'RMSLE': rmsle(y_train, train_pred),
            'R2': r2_score(y_train, train_pred),
            'MAE': mean_absolute_error(y_train, train_pred)
        }
        print(f"{name} Training Time: {duration:.2f}s")
        print("Metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})
    pbar.update(1)
    
    print("\n5. Creating Ensemble")
    ensemble = WeightedEnsemble(list(models.values()))
    ensemble.fit(X_train_selected, y_train)
    print("Model Weights:", {name: f"{w:.4f}" for name, w in zip(models.keys(), ensemble.weights)})
    pbar.update(1)
    
    print("\n6. Making Final Predictions")
    y_pred = ensemble.predict(X_test_selected)
    
    # Calculate cross-validation score
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(KFold(5, shuffle=True, random_state=42).split(X_train_selected)):
        fold_ensemble = WeightedEnsemble(list(models.values()))
        fold_ensemble.fit(X_train_selected[train_idx], y_train[train_idx])
        val_pred = fold_ensemble.predict(X_train_selected[val_idx])
        fold_rmsle = rmsle(y_train[val_idx], val_pred)
        cv_scores.append(fold_rmsle)
        print(f"Fold {fold+1} RMSLE: {fold_rmsle:.4f}")
    
    cv_rmsle = np.mean(cv_scores)
    print(f"\nMean CV RMSLE: {cv_rmsle:.4f} (+/- {np.std(cv_scores)*2:.4f})")
    pbar.update(1)
    
    # Print feature importance summary
    print("\nTop Feature Importance Summary:")
    importance_dfs = []
    for name, model in models.items():
        if hasattr(model, 'feature_importances_') or (hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score')):
            imp_df = analyze_feature_importance(model, feature_cols)
            imp_df['model'] = name
            importance_dfs.append(imp_df)
    
    if importance_dfs:
        combined_importance = pd.concat(importance_dfs)
        print("\nTop 10 Features (Average Importance):")
        print(combined_importance.groupby('feature')['importance'].mean()
              .sort_values(ascending=False).head(10))
    
    total_time = time.time() - start_time
    print(f"\nTotal Training Time: {total_time:.2f}s")
    pbar.close()
    
    return ensemble, y_pred, cv_rmsle, feature_mask

def create_all_features(X, y=None):
    """Create all advanced features including meta-features"""
    features = [X]
    
    # Add polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    features.append(poly.fit_transform(X))
    
    if y is not None:
        # Add meta-features from multiple models
        base_models = [
            lgb.LGBMRegressor(n_estimators=100),
            xgb.XGBRegressor(n_estimators=100),
            GradientBoostingRegressor(n_estimators=100)
        ]
        features.append(create_advanced_meta_features(X, y, base_models))
    
    return np.hstack(features)

def train_lightgbm(X, y):
    """Train LightGBM with optimized splitting criteria"""
    n_workers = get_optimal_workers()
    
    # Force OpenMP settings for LightGBM
    import os
    os.environ['OMP_NUM_THREADS'] = str(n_workers)
    
    params = {
        # Core parameters
        'objective': 'regression',
        'metric': 'rmse',
        'device_type': 'gpu',
        'num_threads': n_workers,
        'force_col_wise': True,
        'verbosity': -1,             # Suppress messages
        
        # Splitting params
        'min_gain_to_split': 0.0,    
        'min_data_in_leaf': 2,       
        'min_sum_hessian_in_leaf': 0,
        'min_data_per_group': 1,     
        
        # Tree structure
        'num_leaves': 31,
        'max_depth': -1,
        
        # Training params
        'learning_rate': 0.1,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'max_bin': 255,
        
        # Performance settings
        'boost_from_average': True,
        'enable_bundle': False,
        'histogram_pool_size': -1
    }
    
    # Create train/val split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create datasets without silent parameter
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    
    print("\nStarting training...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dtrain, dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=10)
        ]
    )
    
    print("\nTraining final model...")
    final_model = lgb.train(
        params,
        lgb.Dataset(X, y, free_raw_data=False),
        num_boost_round=model.best_iteration
    )
    
    return final_model

def train_xgboost(X, y):
    """Train XGBoost with enforced GPU settings using updated parameters"""
    use_gpu = validate_gpu()
    n_workers = get_optimal_workers()
    
    # Define base parameters
    base_params = {
        'n_jobs': n_workers,
        'tree_method': 'hist',
        'device': 'gpu',
        'max_bin': 63 if use_gpu else 255,
        'verbosity': 2,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        # Remove predictor parameter as it's deprecated
    }
    
    # Get optimized parameters
    best_params = optimize_xgb_params(X, y)
    
    # Update with optimized parameters, removing any deprecated ones
    if 'predictor' in best_params:
        del best_params['predictor']
    base_params.update(best_params)
    
    print("\nXGBoost Configuration:")
    print(f"- GPU enabled: {use_gpu}")
    print(f"- Parameters: {base_params}")
    
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = xgb.XGBRegressor(**base_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )
    
    return model

# Helper functions
def softmax(x):
    """Compute softmax values for array x"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def get_optimal_workers():
    """Get optimal number of worker processes"""
    cpu_count = multiprocessing.cpu_count()
    # Leave one core free for system operations
    optimal = max(1, cpu_count - 1)
    return optimal

def validate_gpu():
    """Validate GPU availability and configuration with additional checks"""
    try:
        print("\nChecking GPU configuration:")
        
        # Check system GPU info using nvidia-smi
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            print("nvidia-smi output:\n", result.stdout)
        except Exception as ex:
            print("nvidia-smi check failed:", ex)
        
        # Log library versions for debugging
        print("XGBoost version:", xgb.__version__)
        print("LightGBM version:", lgb.__version__)
        
        # Try XGBoost GPU test with updated parameters
        dummy_data = np.random.rand(100, 10)
        dummy_labels = np.random.rand(100)
        # Updated: use tree_method 'hist' with device='cuda'
        xgb_model = xgb.XGBRegressor(tree_method='hist', device='cuda')
        xgb_model.fit(dummy_data, dummy_labels)
        print("✓ XGBoost GPU working")
        
        # Try LightGBM GPU test
        lgb_model = lgb.LGBMRegressor(device='gpu')
        lgb_model.fit(dummy_data, dummy_labels)
        print("✓ LightGBM GPU working")
        
        return True
    except Exception as e:
        print(f"GPU initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Falling back to CPU")
        return False
