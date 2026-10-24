import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_log_error
import logging
import xgboost as xgb

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_evaluate_model(validate_data_path, model_path, xgboost_path):
    # Load validation data
    data = pd.read_csv(validate_data_path)
    
    # Split data into features and target
    X = data.drop(columns=['PremiumAmount'])
    y = data['PremiumAmount']
    
    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    ### Neural Network ###
    # Load the trained model
    model = load_model(model_path)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Ensure non-negative values for RMSLE calculation
    y = np.maximum(y, 0)
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
    logger.info(f"Validation RMSLE (Neural Network): {rmsle:.4f}")
    
    # Generate submission
    submission = pd.DataFrame({
        'id': data['id'],
        'Premium Amount': y_pred.flatten()
    })
    submission.to_csv("nn_submission.csv", index=False)
    logger.info(f"Created submission file: nn_submission.csv")
    
    ### XGBoost ###
    # Load the trained xgboost model
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgboost_path)
    
    # Make predictions
    dmatrix = xgb.DMatrix(X)
    y_pred = xgb_model.predict(dmatrix)
    
    # Ensure non-negative values for RMSLE calculation
    y_pred = np.maximum(y_pred, 0)
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y, y_pred))
    logger.info(f"Validation RMSLE (XGBoost): {rmsle:.4f}")
    
    # Generate submission
    submission = pd.DataFrame({
        'id': data['id'],
        'Premium Amount': y_pred
    })
    submission.to_csv("xgb_submission.csv", index=False)
    logger.info(f"Created submission file: xgb_submission.csv")

if __name__ == "__main__":
    validate_data_path = "ensembles/intermediate/merged_validate.csv"
    model_path = "ensembles/intermediate/bignet_model.h5"
    xgboost_path = "ensembles/intermediate/xgboost_model.json"
    load_and_evaluate_model(validate_data_path, model_path, xgboost_path)
