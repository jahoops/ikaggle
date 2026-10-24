import pandas as pd
import numpy as np
import os
import sys
import joblib
import time  # Import the time module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager
from utils import feature_cols
from xgboost import XGBRegressor  # Import XGBoost

# Check for GPU availability and provide feedback
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected. Using GPU for training:")
    for gpu in gpus:
        print(f" - {gpu.name}: {gpu.device_type}")
    import cupy as cp  # Import CuPy if GPU is available
else:
    print("No GPU detected. Running on CPU.")
    cp = np  # Use NumPy if no GPU is available

def make_submission():
    # Reload full training data for retraining
    train_data = data_manager.finaltrain.copy()
    print(feature_cols)
    X_train = train_data[feature_cols].copy().values  # Convert to NumPy array
    y_train = train_data["PremiumAmount"].values  # Convert to NumPy array
    
    # Reload test data and extract features (preserving training feature order)
    test_data = data_manager.finaltest.copy()
    X_test = test_data[feature_cols].copy().values  # Convert to NumPy array
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Convert NumPy arrays to CuPy arrays if GPU is available
    X_train = cp.asarray(X_train)
    y_train = cp.asarray(y_train)
    X_test = cp.asarray(X_test)
    
    # Load best model saved from pipeline.py and retrain it
    # Replace SVR with XGBoost
    best_model = XGBRegressor(tree_method='hist', device='cuda', random_state=42)  # Set GPU parameters
    print("Retraining best model on full training data...")
    
    # Measure training time
    start_time = time.time()
    # show length and some content from X_train and y_train
    print(len(X_train))
    print(X_train[:5])
    print(len(y_train))
    print(y_train[:5])
    best_model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Retraining complete. Training time: {training_time:.2f} seconds")
    print(len(X_test))
    print(X_test[:5])
    # Make predictions and clip negatives
    y_pred = best_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)
    
    # Use 'id' column if available; otherwise, fall back to index
    ids = test_data['id'] if 'id' in test_data.columns else test_data.index
    print(f"Using IDs from: {'id column' if 'id' in test_data.columns else 'index'}")
    
    # Create and save submission file
    submission = pd.DataFrame({
        'id': ids,
        'Premium Amount': y_pred
    })
    submission_path = 'best_model_submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

if __name__ == '__main__':
    make_submission()
