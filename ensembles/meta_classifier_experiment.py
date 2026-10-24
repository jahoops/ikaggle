import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_log_error, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_individual_models(train_data, features, target):
    # Split data into features and target
    X = train_data[features]
    y = train_data[target]
    
    # Ensure target values are positive
    y_min = y.min()
    if y_min <= 0:
        y += abs(y_min) + 1
    
    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Define and train individual models
    models = {}
    
    # XGBoost model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    print('xgb fit')
    xgb_model.fit(X, y)
    models['xgboost'] = xgb_model
    
    # Neural Network model
    def create_nn_model():
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_logarithmic_error')
        return model
    
    nn_model = create_nn_model()
    nn_model.fit(X, y, epochs=1, batch_size=32, verbose=1)
    models['neuralnet'] = nn_model
    
    return models, scaler

def generate_predictions(models, scaler, data, features):
    X = data[features]
    X = scaler.transform(X)
    
    predictions = {}
    for model_name, model in models.items():
        if model_name == 'neuralnet':
            predictions[model_name] = model.predict(X).flatten()
        else:
            predictions[model_name] = model.predict(X)
    
    return predictions

def train_meta_classifier(train_data, predictions, target):
    # Create meta features
    meta_features = pd.DataFrame(predictions)
    meta_features['target'] = train_data[target]
    
    # Train a meta-classifier
    X_meta = meta_features.drop(columns=['target'])
    y_meta = meta_features['target']
    
    # Optimize RandomForestClassifier to use fewer resources
    meta_classifier = RandomForestClassifier(
        n_estimators=50,       # Reduce the number of trees
        max_depth=10,          # Limit tree depth
        max_features='sqrt',   # Use square root of the number of features
        max_samples=0.8,       # Use 80% of the data for each tree
        n_jobs=12              # Use all available CPU cores
    )

    print("classifier fit")
    meta_classifier.fit(X_meta, y_meta)
    
    return meta_classifier

def identify_outliers(meta_classifier, predictions):
    # Predict which model to use
    meta_features = pd.DataFrame(predictions)

    print("outlier predict")
    model_predictions = meta_classifier.predict(meta_features)
    
    # Identify outliers
    outliers = meta_features[model_predictions == -1]
    
    return model_predictions, outliers

def run_meta_classifier_experiment(train_data, validate_data, test_data):
    # Define the features and target
    features = [
        'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 'PreviousClaims', 'VehicleAge', 
        'CreditScore', 'InsuranceDuration', 'SignupYear', 'SignupMonth', 'MaritalStatus_Married', 
        'EducationLevel_PhD', 'Occupation_SelfEmployed', 'Occupation_nan', 'Location_Rural', 
        'Location_Urban', 'CustomerFeedback_Average', 'CustomerFeedback_Good', 'SmokingStatus_No', 
        'ExerciseFrequency_Daily', 'PropertyType_Apartment', 'PropertyType_House'
    ]
    target = 'PremiumAmount'
    
    # Train individual models
    models, scaler = train_individual_models(train_data, features, target)
    
    # Generate predictions for training and validation data
    train_predictions = generate_predictions(models, scaler, train_data, features)
    validate_predictions = generate_predictions(models, scaler, validate_data, features)
    
    # Train meta-classifier
    meta_classifier = train_meta_classifier(train_data, train_predictions, target)
    
    # Identify outliers in validation data
    model_predictions, outliers = identify_outliers(meta_classifier, validate_predictions)
    
    # Evaluate meta-classifier
    y_validate = validate_data[target]
    print("evaluating rmsle")
    rmsle = np.sqrt(mean_squared_log_error(y_validate, model_predictions))
    logger.info(f"Validation RMSLE: {rmsle:.4f}")
    
    # Make predictions on test data
    print("test predictions")
    test_predictions = generate_predictions(models, scaler, test_data, features)
    test_model_predictions, test_outliers = identify_outliers(meta_classifier, test_predictions)
    
    # Save test predictions
    print("save test predictions")
    test_data['PremiumAmount'] = test_model_predictions
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
    test = pd.read_csv(data_dir / "merged_test.csv")  # Assuming test data is in test.csv
    run_meta_classifier_experiment(train, validate, test)
