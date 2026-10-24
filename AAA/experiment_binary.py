import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tensorflow import keras
import optuna
from datetime import datetime
import json
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def custom_rmsle(y_true, y_pred):
    """Calculate RMSLE with proper handling of negative values"""
    # Clip predictions to minimum of 1 to avoid log(0)
    y_pred = np.clip(y_pred, 1, None)
    # Clip true values to minimum of 1
    y_true = np.clip(y_true, 1, None)
    return np.sqrt(np.mean(np.power(np.log(y_pred) - np.log(y_true), 2)))

class BinaryExperiment:
    def __init__(self, data_dir: Path):
        # Set TensorFlow configuration at startup
        if 'tensorflow' not in sys.modules:
            import tensorflow as tf
            # Configure TensorFlow before any operations
            tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count() - 1)
            tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count() - 1)
        
        self.data_dir = data_dir
        self.output_dir = data_dir.parent / 'experiments'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        print("Loading binary data...")
        self.train = pd.read_csv(data_dir / 'train_bitbins.csv')
        self.test = pd.read_csv(data_dir / 'test_bitbins.csv')
        
        # Split features and target
        self.target = 'PremiumAmount'
        self.features = [col for col in self.train.columns 
                        if col not in [self.target, 'id']]
        
        print(f"Loaded {len(self.features)} binary features")
        
        # Check GPU availability
        self.use_gpu = self._check_gpu()
        self.n_jobs = os.cpu_count() - 1  # Leave one core free
        print(f"GPU available: {self.use_gpu}")
        print(f"Using {self.n_jobs} CPU cores")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import tensorflow as tf
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_available:
                # Set memory growth to avoid taking all GPU memory
                for device in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(device, True)
            return gpu_available
        except:
            return False

    def create_neural_network(self, trial: optuna.Trial) -> keras.Model:
        """Create neural network with trial parameters"""
        n_layers = trial.suggest_int('n_layers', 2, 4)
        n_units = trial.suggest_int('n_units', 64, 512)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        if self.use_gpu:
            # Use mixed precision for better GPU performance
            keras.mixed_precision.set_global_policy('mixed_float16')
        
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(len(self.features),)))
        
        for i in range(n_layers):
            model.add(keras.layers.Dense(
                n_units // (2 ** i),
                activation='relu'
            ))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout))
        
        model.add(keras.layers.Dense(1))
        model.compile(
            optimizer=keras.optimizers.Adam(
                trial.suggest_float('learning_rate', 1e-4, 1e-2)
            ),
            loss='mse'
        )
        
        return model
    
    def optimize_lightgbm_trial(self, trial: optuna.Trial, X, y) -> float:
        """Single trial for LightGBM optimization"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'n_jobs': self.n_jobs
        }
        
        if self.use_gpu:
            params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'max_bin': 63
            })
        
        # Use cross-validation for more robust optimization
        cv_scores = []
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = model.predict(X.iloc[val_idx])
            score = custom_rmsle(y.iloc[val_idx], pred)  # Use custom_rmsle
            cv_scores.append(score)
        
        return np.mean(cv_scores)

    def get_default_params(self, model_type: str) -> Dict:
        """Get default parameters if optimization fails"""
        if model_type == 'lightgbm':
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'min_child_samples': 20,
                'n_jobs': self.n_jobs
            }
            if self.use_gpu:
                params.update({
                    'device': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'max_bin': 63
                })
            return params
        # Add other model defaults as needed
        return {}

    def train_lightgbm(self, kf: KFold) -> Tuple[np.ndarray, float]:
        """Train LightGBM model with robust optimization"""
        try:
            # Setup optimization study
            study = optuna.create_study(direction='minimize')
            
            # Get training data for optimization
            X = self.train[self.features]
            y = self.train[self.target]
            
            # Run optimization with timeout
            study.optimize(
                lambda trial: self.optimize_lightgbm_trial(trial, X, y),
                n_trials=10,
                timeout=300  # 5 minutes timeout
            )
            
            # Use best params or fallback to defaults
            if study.best_params:
                params = {**self.get_default_params('lightgbm'), **study.best_params}
            else:
                print("Warning: Optimization failed, using default parameters")
                params = self.get_default_params('lightgbm')
            
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            print("Using default parameters")
            params = self.get_default_params('lightgbm')
        
        # Train and evaluate
        cv_scores = []
        test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train)):
            print(f"Training fold {fold + 1}/5...")
            
            # Get fold data
            X_train = self.train.loc[train_idx, self.features]
            y_train = self.train.loc[train_idx, self.target]
            X_val = self.train.loc[val_idx, self.features]
            y_val = self.train.loc[val_idx, self.target]
            
            # Train model with correct parameters
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # Disable logging
                ]
            )
            
            # Evaluate using custom RMSLE
            pred = model.predict(X_val)
            score = custom_rmsle(y_val, pred)  # Use custom_rmsle
            cv_scores.append(score)
            
            # Predict test using best iteration
            test_preds.append(
                model.predict(
                    self.test[self.features],
                    num_iteration=model.best_iteration_
                )
            )
            
            print(f"Fold {fold + 1} score: {score:.4f}")
        
        mean_cv = np.mean(cv_scores)
        print(f"\nLightGBM CV score: {mean_cv:.4f} (+/- {np.std(cv_scores):.4f})")
        
        return np.mean(test_preds, axis=0), mean_cv

    def train_and_evaluate(self) -> Tuple[float, pd.DataFrame]:
        """Train models and create submission"""
        results = []
        predictions = []
        
        # Setup cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Train multiple models
        for model_name, model_func in [
            ('lightgbm', self.train_lightgbm),
            ('neural_net', self.train_neural_net),
            ('catboost', self.train_catboost)
        ]:
            print(f"\nTraining {model_name}...")
            model_preds, cv_score = model_func(kf)
            results.append({
                'model': model_name,
                'cv_score': cv_score
            })
            predictions.append(model_preds)
        
        # Create ensemble prediction
        final_pred = np.mean(predictions, axis=0)
        
        # Create submission
        submission = pd.DataFrame({
            'id': self.test['id'],
            'PremiumAmount': final_pred
        })
        
        # Save results
        results_df = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        submission.to_csv(
            self.output_dir / f'submission_binary_{timestamp}.csv',
            index=False
        )
        results_df.to_csv(
            self.output_dir / f'model_results_{timestamp}.csv',
            index=False
        )
        
        return results_df['cv_score'].mean(), submission
    
    def train_neural_net(self, kf: KFold) -> Tuple[np.ndarray, float]:
        """Train neural network model"""
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.optimize_neural_net(trial), n_trials=10)
        
        cv_scores = []
        test_preds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train)):
            X_train = self.train.loc[train_idx, self.features].values
            y_train = self.train.loc[train_idx, self.target].values
            X_val = self.train.loc[val_idx, self.features].values
            y_val = self.train.loc[val_idx, self.target].values
            
            model = self.create_neural_network(study.best_trial)
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
            pred = model.predict(X_val).flatten()
            score = custom_rmsle(y_val, pred)  # Use custom_rmsle
            cv_scores.append(score)
            
            test_preds.append(model.predict(self.test[self.features]).flatten())
        
        return np.mean(test_preds, axis=0), np.mean(cv_scores)
    
    def optimize_neural_net(self, trial: optuna.Trial) -> float:
        """Optimize neural network hyperparameters"""
        # Get training data for optimization
        X = self.train[self.features].values
        y = self.train[self.target].values
        
        # Create model with trial parameters
        model = self.create_neural_network(trial)
        
        # Use cross-validation for robust optimization
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,  # Reduced epochs for optimization
                batch_size=32,
                verbose=0
            )
            
            # Evaluate
            pred = model.predict(X_val).flatten()
            score = custom_rmsle(y_val, pred)  # Use custom_rmsle
            scores.append(score)
            
            # Clean up to avoid memory issues
            keras.backend.clear_session()
        
        return np.mean(scores)
    
    def train_catboost(self, kf: KFold) -> Tuple[np.ndarray, float]:
        """Train CatBoost model with GPU support"""
        cv_scores = []
        test_preds = []
        
        params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False,
            'thread_count': self.n_jobs
        }
        
        if self.use_gpu:
            params.update({
                'task_type': 'GPU',
                'devices': '0',
            })
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.train)):
            X_train = self.train.loc[train_idx, self.features]
            y_train = self.train.loc[train_idx, self.target]
            X_val = self.train.loc[val_idx, self.features]
            y_val = self.train.loc[val_idx, self.target]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train)
            
            pred = model.predict(X_val)
            score = custom_rmsle(y_val, pred)  # Use custom_rmsle
            cv_scores.append(score)
            
            test_preds.append(model.predict(self.test[self.features]))
        
        return np.mean(test_preds, axis=0), np.mean(cv_scores)

if __name__ == "__main__":
    data_dir = Path('/home/USER/Projects/ikaggle/AAA/bitbins')
    experiment = BinaryExperiment(data_dir)
    cv_score, submission = experiment.train_and_evaluate()
    
    print(f"\nExperiment complete!")
    print(f"Mean CV Score: {cv_score:.4f}")
    print(f"Results saved to: {experiment.output_dir}")
