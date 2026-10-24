import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from utils import safe_rmsle

logger = logging.getLogger(__name__)

def get_latest_model_files(model_dirs):
    """Get latest PKL file from each model output directory"""
    latest_models = {}
    for name, directory in model_dirs.items():
        try:
            files = list(Path(directory).glob('best_*_model_*.pkl'))
            if files:
                latest = max(files, key=lambda x: x.stat().st_mtime)
                latest_models[name] = latest
                logger.info(f"Found {name} model: {latest.name}")
        except Exception as e:
            logger.warning(f"No model found for {name}: {e}")
    
    return latest_models

def get_model_specific_features(model_name, data):
    """Get correct feature set for each model"""
    base_features = ['HealthScore', 'AnnualIncome', 'CreditScore', 'Age']
    
    if model_name in ['Catboost', 'Knn']:
        dist_cols = [f'DIST{i}' for i in range(1, 21)]
        return ['id'] + base_features + dist_cols
        
    elif model_name == 'Lightgbm':
        cat_features = [
            'CustomerFeedback_Average', 'CustomerFeedback_Good',
            'EducationLevel_PhD', 'ExerciseFrequency_Daily',
            'InsuranceDuration', 'Location_Rural', 'Location_Urban',
            'MaritalStatus_Married', 'NumberofDependents',
            'Occupation_SelfEmployed', 'Occupation_nan', 'PreviousClaims',
            'PropertyType_Apartment', 'PropertyType_House',
            'SignupMonth', 'SignupYear', 'SmokingStatus_No',
            'VehicleAge'
        ]
        return ['id'] + base_features + cat_features
        
    else:
        return ['id'] + base_features

def load_model_data(base_path, nn_path):
    """Load and prepare data with all required features"""
    # Load base data
    data = pd.read_csv(base_path)
    
    # Add distance features
    nn_data = pd.read_csv(nn_path)
    data = pd.merge(data, nn_data, on='id', how='left')
    
    # Create distance features
    dist_cols = [f'DIST{i}' for i in range(1, 21)]
    data['avg_dist'] = data[dist_cols].mean(axis=1)
    data['min_dist'] = data[dist_cols].min(axis=1)
    
    return data

def get_active_venv():
    """Detect active virtual environment"""
    return os.environ.get('VIRTUAL_ENV', '').split('/')[-1]

def is_tf_environment():
    """Check if current environment supports TensorFlow"""
    return get_active_venv() == 'vnext'

def get_compatible_models(model_dirs):
    """Filter models based on environment"""
    tf_models = ['Xgboost', 'Neuralnet']
    compatible_dirs = {}
    
    for name, directory in model_dirs.items():
        if name in tf_models and not is_tf_environment():
            continue
        compatible_dirs[name] = directory
    
    return compatible_dirs

def get_model_requirements(model):
    """Extract feature requirements from model metadata"""
    if hasattr(model, 'feature_cols_in_'):
        return list(model.feature_cols_in_)
    return ['HealthScore', 'AnnualIncome', 'CreditScore', 'Age']

def generate_predictions(model, data, model_name=None):
    if hasattr(model, 'feature_cols_in_'):
        needed_cols = list(model.feature_cols_in_)
    else:
        needed_cols = [...]  # fallback if not stored
    
    X = data.reindex(columns=needed_cols)
    return model.predict(X)

def generate_submission():
    # Get models compatible with current environment
    model_dirs = {
        'Catboost': 'output/Catboost',
        'Knn': 'output/Knn',
        'Lightgbm': 'output/Lightgbm',
        'Linear': 'output/Linear',
        'Neuralnet': 'output/Neuralnet',
        'Randomforest': 'output/Randomforest',
        'Ridge': 'output/Ridge',
        # 'SVR': 'output/SVR',
        'Xgboost': 'output/Xgboost'
    }
    
    compatible_dirs = get_compatible_models(model_dirs)
    model_files = get_latest_model_files(compatible_dirs)
    
    # Load data
    train_data = load_model_data(
        "data/processed/feature_train.csv",
        "data/processed/nn_train.csv"
    )
    test_data = load_model_data(
        "data/processed/feature_test.csv", 
        "data/processed/nn_test.csv"
    )
    
    # Get model predictions
    train_predictions = {}
    test_predictions = {}
    model_names = []
    
    for name, model_path in model_files.items():
        try:
            model = joblib.load(model_path)
            train_pred = generate_predictions(model, train_data, name)
            test_pred = generate_predictions(model, test_data, name)
            
            if train_pred is not None and test_pred is not None:
                train_predictions[name] = train_pred
                test_predictions[name] = test_pred
                logger.info(f"Generated predictions for {name}")
        except Exception as e:
            logger.error(f"Failed to process {name}: {e}")
    
    if not train_predictions:
        logger.error("No valid predictions found")
        return
    
    # Stack predictions for ensemble
    X_train = np.column_stack(list(train_predictions.values()))
    y_train = train_data['PremiumAmount']
    X_test = np.column_stack(list(test_predictions.values()))
    
    # Train ensemble
    ensemble = RandomForestRegressor(n_estimators=100, random_state=42)
    ensemble.fit(X_train, y_train)
    
    # Generate submission
    final_predictions = ensemble.predict(X_test)
    submission = pd.DataFrame({
        'id': test_data['id'],
        'PremiumAmount': final_predictions
    })
    
    submission.to_csv("submission.csv", index=False)
    logger.info(f"Created submission using {len(model_names)} models: {', '.join(model_names)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_submission()