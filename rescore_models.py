import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import logging

# Constants
RECORD_LIMIT = 100000

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_experiment_state(state_path):
    """Load experiment_state.json"""
    with open(state_path, 'r') as f:
        state = json.load(f)
    return state

def save_experiment_state(state_path, state):
    """Save updated experiment_state.json"""
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"Updated experiment state saved to {state_path}")

def compute_rmsle(y_true, y_pred):
    """Compute Root Mean Squared Logarithmic Error"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def rescore_models(output_dir):
    """Rescore models and update experiment_state.json using sampled data"""
    output_v1 = Path(output_dir) / 'v1'
    state_path = output_v1 / 'experiment_state.json'
    train_path = Path(output_dir) / 'cleaned_train.csv'
    
    # Load and sample training data
    train_data = pd.read_csv(train_path)
    if len(train_data) > RECORD_LIMIT:
        train_data = train_data.sample(n=RECORD_LIMIT, random_state=42)
        logger.info(f"Sampled {RECORD_LIMIT} records from training data")
    
    X_train = train_data.drop(columns=['PremiumAmount'])
    y_train = train_data['PremiumAmount']
    
    # Load experiment state
    state = load_experiment_state(state_path)
    
    for model_info in state['completed']:
        model_name = model_info['name']
        model_path = output_v1 / f'best_{model_name}_model.pkl'
        if not model_path.exists():
            logger.warning(f"Model file {model_path} does not exist. Skipping.")
            continue
        try:
            loaded = joblib.load(model_path)
            model = loaded['model']
            features = loaded.get('features')
            if not features:
                logger.warning(f"No features found for model {model_name}. Skipping.")
                continue
            X = X_train[features]
            predictions = model.predict(X)
            rmsle = compute_rmsle(y_train, predictions)
            logger.info(f"Model: {model_name}, RMSLE: {rmsle:.4f}")
            model_info['score'] = rmsle
        except Exception as e:
            logger.error(f"Error rescoring model {model_name}: {str(e)}")
            model_info['score'] = None
            model_info['status'] = 'rescore_failed'
    
    # Save updated state
    save_experiment_state(state_path, state)
    
if __name__ == "__main__":
    output_directory = Path('/home/USER/Kaggle/InsuranceNext/output')
    rescore_models(output_directory)