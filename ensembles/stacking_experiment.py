import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(train, validate, test):
    runStacking(train, validate, test)

def runStacking(train_data, validate_data, test_data):
    # Define the features and target
    features = [
        'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 'PreviousClaims', 'VehicleAge', 
        'CreditScore', 'InsuranceDuration', 'SignupYear', 'SignupMonth', 'MaritalStatus_Married', 
        'EducationLevel_PhD', 'Occupation_SelfEmployed', 'Occupation_nan', 'Location_Rural', 
        'Location_Urban', 'CustomerFeedback_Average', 'CustomerFeedback_Good', 'SmokingStatus_No', 
        'ExerciseFrequency_Daily', 'PropertyType_Apartment', 'PropertyType_House', 
        'catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 
        'svr', 'xgboost'
    ]
    target = 'PremiumAmount'
    
    # Split data into features and target
    X_train = train_data[features]
    y_train = train_data[target]
    X_validate = validate_data[features]
    y_validate = validate_data[target]
    X_test = test_data[features]
    
    # Ensure target values are positive
    y_min = y_train.min()
    if y_min <= 0:
        y_train += abs(y_min) + 1
        y_validate += abs(y_min) + 1
    
    # Standardize the features
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_validate = scaler.transform(X_validate)
    X_test = scaler.transform(X_test)
    
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
    model.fit(X_train, y_train, eval_set=[(X_validate, y_validate)], verbose=True)
    
    # Evaluate the model
    y_pred_validate = model.predict(X_validate)
    
    # Ensure predictions are positive
    y_pred_validate = np.maximum(y_pred_validate, 0)
    
    rmsle = np.sqrt(mean_squared_log_error(y_validate, y_pred_validate))
    logger.info(f"Validation RMSLE: {rmsle:.4f}")
    
    # Make predictions on the test data
    y_pred_test = model.predict(X_test)
    y_pred_test = np.maximum(y_pred_test, 0)
    
    # Save the final model
    final_model_path = 'ensembles/intermediate/stacking_model.json'
    model.save_model(final_model_path)
    logger.info(f"Model saved to {final_model_path}")
    
    # Save the test predictions
    test_data['PremiumAmount'] = y_pred_test
    test_predictions_path = 'ensembles/intermediate/test_predictions.csv'
    test_data[['id', 'PremiumAmount']].to_csv(test_predictions_path, index=False)
    logger.info(f"Test predictions saved to {test_predictions_path}")

    # Print the current score
    print(f"Validation RMSLE score: {rmsle:.4f}")

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")
    train = pd.read_csv(data_dir / "merged_train.csv")
    validate = pd.read_csv(data_dir / "merged_validate.csv")
    test = pd.read_csv(data_dir / "test.csv")  # Assuming test data is in test.csv
    start(train, validate, test)
