import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split
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

def improve_poorly_estimated(train_data, validate_data, poorly_estimated_data):
    # Define the features and target
    features = [
        'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 'PreviousClaims', 'VehicleAge', 
        'CreditScore', 'InsuranceDuration', 'SignupYear', 'SignupMonth', 'MaritalStatus_Married', 
        'EducationLevel_PhD', 'Occupation_SelfEmployed', 'Occupation_nan', 'Location_Rural', 
        'Location_Urban', 'CustomerFeedback_Average', 'CustomerFeedback_Good', 'SmokingStatus_No', 
        'ExerciseFrequency_Daily', 'PropertyType_Apartment', 'PropertyType_House', 'avg_dist', 
        'median_dist', 'avg_pred', 'median_pred', 'catboost', 'knn', 'lasso', 'lightgbm', 'linear', 
        'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost'
    ]
    target = 'PremiumAmount'
    
    # Split data into features and target
    X_train = poorly_estimated_data[features]
    y_train = poorly_estimated_data[target]
    X_validate = validate_data[features]
    y_validate = validate_data[target]
    
    # Ensure target values are positive
    y_min = y_train.min()
    if y_min <= 0:
        y_train += abs(y_min) + 1
        y_validate += abs(y_min) + 1
    
    # Standardize the features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)
    
    # Define the XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    # Train the model
    logger.info("Starting model training...")
    model.fit(X_train, y_train, verbose=True)
    
    # Evaluate the model on the validation set
    y_pred = model.predict(X_validate)
    
    # Ensure predictions are positive
    y_pred = np.maximum(y_pred, 0)
    
    # Replace the predictions for the poorly estimated rows
    validate_data.loc[validate_data['id'].isin(poorly_estimated_data['id']), 'PremiumAmount'] = y_pred
    
    # Calculate RMSLE
    rmsle = np.sqrt(mean_squared_log_error(y_validate, validate_data['PremiumAmount']))
    logger.info(f"Validation RMSLE after improvement: {rmsle:.4f}")
    
    # Save the improved validation set
    improved_validate_path = 'ensembles/intermediate/improved_validate.csv'
    validate_data.to_csv(improved_validate_path, index=False)
    logger.info(f"Improved validation set saved to {improved_validate_path}")

    # Print the current score
    print(f"Validation RMSLE score after improvement: {rmsle:.4f}")

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")
    train = pd.read_csv(data_dir / "merged_train.csv")
    validate = pd.read_csv(data_dir / "merged_validate.csv")
    poorly_estimated = pd.read_csv(data_dir / "poorly_estimated_rows_xgboost.csv")
    improve_poorly_estimated(train, validate, poorly_estimated)
