from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
# Add parent directory to sys.path
import numpy as np
import os
import sys
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager
from utils import feature_cols

# Define RMSLE scorer with clipping for negative values
def rmsle(y_true, y_pred):
    y_true_clip = np.maximum(y_true, 0)
    y_pred_clip = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true_clip, y_pred_clip))

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

data = data_manager.finaltrain.copy()

# Load your data
X = data[feature_cols]
y = data['PremiumAmount']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines for different models
xgb_pipeline1 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('xgb', XGBRegressor(tree_method='gpu_hist', random_state=42))
])

xgb_pipeline2 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('xgb', XGBRegressor(tree_method='gpu_hist', random_state=42))
])

svr_pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, interaction_only=True)),
    ('svr', SVR())
])

# Add TPOT's successful stacked configuration
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator

tpot_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.01, max_depth=3, 
        n_estimators=100, random_state=42, subsample=0.8, 
        tree_method='hist',device='cuda'
    )),
    StackingEstimator(estimator=XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.01, max_depth=5, 
        n_estimators=100, random_state=42, subsample=0.8, 
        tree_method='hist',device='cuda'
    )),
    XGBRegressor(
        colsample_bytree=0.8, learning_rate=0.1, max_depth=3, 
        n_estimators=100, random_state=42, subsample=0.8, 
        tree_method='hist',device='cuda'
    )
)

# Define parameter grids for each model
xgb_param_grid1 = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__subsample': [0.8, 0.9],
    'xgb__colsample_bytree': [0.8, 0.9]
}

xgb_param_grid2 = {
    'xgb__n_estimators': [50, 150],
    'xgb__max_depth': [4, 6],
    'xgb__learning_rate': [0.05, 0.2],
    'xgb__subsample': [0.7, 0.85],
    'xgb__colsample_bytree': [0.7, 0.85]
}

svr_param_grid = {
    'svr__C': [0.1, 1, 10],
    'svr__epsilon': [0.01, 0.1, 0.5],
    'svr__kernel': ['linear', 'poly', 'rbf']
}

# Combine pipelines and parameter grids
pipelines = [
    (tpot_pipeline, {}),  # Empty param_grid since we're using TPOT's optimized parameters
    (xgb_pipeline1, xgb_param_grid1),
    (xgb_pipeline2, xgb_param_grid2),
    (svr_pipeline, svr_param_grid)
]

from joblib import Parallel, parallel_backend, Memory
import tempfile
import time
import glob

# Add checkpoint functionality
checkpoint_dir = "pipeline_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(pipeline_idx, batch_idx, model, score):
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_pipe{pipeline_idx}_batch{batch_idx}.joblib")
    checkpoint_data = {
        'pipeline_idx': pipeline_idx,
        'batch_idx': batch_idx,
        'model': model,
        'score': score,
        'best_models': best_models
    }
    joblib.dump(checkpoint_data, checkpoint_file)
    print(f"Checkpoint saved: {checkpoint_file}")

def load_latest_checkpoint():
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_pipe*.joblib"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Loading checkpoint: {latest_checkpoint}")
    return joblib.load(latest_checkpoint)

# Modify training loop to use checkpoints
best_models = {}
checkpoint_data = load_latest_checkpoint()

start_pipeline_idx = 0
start_batch_idx = 0

if checkpoint_data:
    start_pipeline_idx = checkpoint_data['pipeline_idx']
    start_batch_idx = checkpoint_data['batch_idx'] + 1
    best_models = checkpoint_data['best_models']
    print(f"Resuming from pipeline {start_pipeline_idx}, batch {start_batch_idx}")

best_models = {}
iteration_count = 0
total_iterations = sum([len(param_grid) for _, param_grid in pipelines]) * 3  # Estimate total iterations

# Add memory monitoring
import psutil
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

# Modify training parameters for stability
batch_size = 500  # Reduced from 1000
num_batches = len(X_train) // batch_size + 1

# Reduce parallel processing load
n_jobs = 2  # Reduced from 5
cv_folds = 3  # Reduced from 5

import itertools

def param_grid_size(grid):
    # Compute total combinations in the grid
    counts = [len(v) for v in grid.values()] if grid else [0]
    size = np.prod(counts)
    return size

# Modify training loop to dynamically set n_iter
for pipeline_idx, (pipeline, param_grid) in enumerate(pipelines[start_pipeline_idx:], start_pipeline_idx):
    for batch_idx in range(start_batch_idx, num_batches):
        try:
            print(f"Memory usage before batch: {get_memory_usage():.2f} MB")
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(X_train))
            X_batch = X_train[start_idx:end_idx].copy()  # Making an explicit copy
            y_batch = y_train[start_idx:end_idx].copy()
            
            # Dynamically adjust n_iter based on grid size to avoid warnings
            grid_size = param_grid_size(param_grid)
            adjusted_n_iter = min(3, grid_size) if grid_size > 0 else 3
            if adjusted_n_iter < 3:
                print(f"Warning: Parameter grid size ({grid_size}) is smaller than desired n_iter=3. Using {adjusted_n_iter} iteration(s).")
            
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=adjusted_n_iter,
                scoring=rmsle_scorer,
                cv=cv_folds,
                random_state=42,
                n_jobs=n_jobs,
                verbose=10,
                error_score='raise'
            )
            
            # Fit using the current batch
            search.fit(X_batch, y_batch)
            
            best_models[pipeline] = search.best_estimator_
            save_checkpoint(pipeline_idx, batch_idx, search.best_estimator_, search.best_score_)
            
            # Force garbage collection
            import gc
            gc.collect()
            print(f"Memory usage after batch: {get_memory_usage():.2f} MB")
            
        except Exception as e:
            print(f"Error in pipeline {pipeline_idx}, batch {batch_idx}: {str(e)}")
            print(f"Memory usage at error: {get_memory_usage():.2f} MB")
            save_checkpoint(pipeline_idx, batch_idx, None, float('inf'))
            continue

# Evaluate the best models on the test set
results = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    score = rmsle(y_test, y_pred)
    results[model_name] = score
    print(f"Model: {model_name}, RMSLE: {score}")

# Print out the best model and its hyperparameters
best_model_name = min(results, key=results.get)
print(f"Best model: {best_model_name}")
print(f"Best hyperparameters: {best_models[best_model_name].get_params()}")

# Save the best model and its parameters
best_model = best_models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')
print("Best model saved as 'best_model.pkl'")