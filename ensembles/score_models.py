import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import os
import sys
import joblib

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from utils import load_data, safe_rmsle
from experiments.default_features import feature_importances
from experiments.models import (
    SVRExperiment, RandomForestExperiment, LightGBMExperiment,
    CatBoostExperiment, KNNExperiment, LinearRegressionExperiment,
    RidgeRegressionExperiment, XGBoostExperiment, NeuralNetworkExperiment
)

logger = logging.getLogger(__name__)

def get_active_venv():
    """Detect active virtual environment"""
    return os.environ.get('VIRTUAL_ENV', '').split('/')[-1]

def is_tf_environment():
    """Check if current environment supports TensorFlow"""
    return get_active_venv() == 'vnext'

def load_trained_model(model_name):
    """Load a trained model from output directory"""
    model_path = Path(f"ensemble/input/best_{model_name}_model_*.pkl")
    try:
        model_file = list(Path().glob(str(model_path)))[0]
        return joblib.load(model_file)
    except (IndexError, IOError) as e:
        logger.error(f"Could not load model for {model_name}: {e}")
        return None

def score_model(model_name, data_path, output_path):
    try:
        model = load_trained_model(model_name)
        data = load_data(data_path)
        
        # Core features
        features = ['HealthScore', 'AnnualIncome', 'CreditScore', 'Age']
        X = data[features]
        ids = data['id']
        
        predictions = model.predict(X)
        
        results = pd.DataFrame({
            'id': ids,
            'prediction': predictions,
            'model': model_name
        })
        if 'PremiumAmount' in data.columns:
            results['PremiumAmount'] = data['PremiumAmount']
            
        results.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error scoring {model_name}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    current_env = get_active_venv()
    logger.info(f"Running in environment: {current_env}")
    
    # Separate models by environment
    tf_models = [
        (XGBoostExperiment, "XGBoost"),
        (NeuralNetworkExperiment, "NeuralNet")
    ]
    
    basic_models = [
        (SVRExperiment, "SVR"),
        (RandomForestExperiment, "RandomForest"),
        (LightGBMExperiment, "LightGBM"),
        (CatBoostExperiment, "CatBoost"),
        (KNNExperiment, "KNN"),
        (LinearRegressionExperiment, "LinearRegression"),
        (RidgeRegressionExperiment, "Ridge")
    ]
    
    models = tf_models if is_tf_environment() else basic_models
    logger.info(f"Will run models: {[m[1] for m in models]}")
    
    for experiment_class, model_name in models:
        # Check train data
        train_path = Path("data/processed/f_train.csv")
        if not train_path.exists():
            logger.error(f"Training data not found at {train_path}")
            continue
            
        # Score training data
        score_model(model_name, str(train_path), 
                   f"ensemble/output/train_scores_{model_name}.csv")
        
        # Check validation data
        val_path = Path("data/processed/feature_validate.csv")
        if val_path.exists():
            score_model(model_name, str(val_path),
                       f"ensemble/output/validation_scores_{model_name}.csv")
        else:
            logger.info(f"Validation data not found at {val_path}, skipping validation scoring")