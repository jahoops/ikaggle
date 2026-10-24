import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(train):
    runBoosting(train)

def runBoosting(data):
    # Split data into features and target
    X = data.drop(columns=['PremiumAmount'])
    y = data['PremiumAmount']
    
    # Ensure target values are positive
    y_min = y.min()
    if y_min <= 0:
        y += abs(y_min) + 1
    
    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Define the XGBoost model
    def create_model():
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        return model
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []
    poorly_estimated_rows = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = create_model()
        
        # Train the model with early stopping
        logger.info("Starting model training...")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
        
        # Evaluate the model
        y_pred = model.predict(X_val)
        
        # Ensure predictions are positive
        y_pred = np.maximum(y_pred, 0)
        
        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        rmsle_scores.append(rmsle)
        logger.info(f"Fold Validation RMSLE: {rmsle:.4f}")
        
        # Identify poorly estimated rows
        abs_diff = np.abs(y_val - y_pred)
        threshold = np.percentile(abs_diff, 90)  # Top 10% worst predictions
        poorly_estimated = data.iloc[val_index][abs_diff > threshold]
        poorly_estimated_rows.append(poorly_estimated)
    
    # Calculate average RMSLE across folds
    avg_rmsle = np.mean(rmsle_scores)
    logger.info(f"Average Validation RMSLE: {avg_rmsle:.4f}")
    
    # Save the final model
    final_model_path = 'ensembles/intermediate/xgboost_model.json'
    model.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Save poorly estimated rows for further analysis
    poorly_estimated_df = pd.concat(poorly_estimated_rows)
    poorly_estimated_df.to_csv('ensembles/intermediate/poorly_estimated_rows_xgboost.csv', index=False)
    logger.info("Saved poorly estimated rows to ensembles/intermediate/poorly_estimated_rows_xgboost.csv")

    # Print the current score
    print(f"Average RMSLE score: {avg_rmsle:.4f}")

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    train = pd.read_csv(f"ensembles/intermediate/merged_train.csv")
    start(train)
