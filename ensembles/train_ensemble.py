import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from utils import load_data, safe_rmsle
from experiments.default_features import feature_importances

feature_importances = [
    "HealthScore", "AnnualIncome", "CreditScore", "Age", "SignupYear",
    "PreviousClaims", "NumberofDependents", "Location_Rural", "VehicleAge",
    "InsuranceDuration", "CustomerFeedback_Average", "SmokingStatus_No",
    "PropertyType_House", "SignupMonth", "Occupation_SelfEmployed",
    "MaritalStatus_Married", "Occupation_nan", "CustomerFeedback_Good",
    "EducationLevel_PhD", "PropertyType_Apartment", "Location_Urban",
    "ExerciseFrequency_Daily"
]

def load_scores(file_paths):
    dfs = [pd.read_csv(file) for file in file_paths]
    return pd.concat(dfs, axis=0)

def train_ensemble():
    # Load models with predictions
    models = {}
    for model_file in Path('ensemble/input').glob('best_*_model_*.pkl'):
        model = joblib.load(model_file)
        models[model.metadata['name']] = model
    
    # Create feature matrix
    X = pd.DataFrame()
    first_model = next(iter(models.values()))
    train_ids = first_model.predictions['train']['id']
    
    # Add core features from first model's metadata
    train_data = load_data("data/processed/feature_train.csv")
    X[first_model.feature_cols] = train_data[first_model.feature_cols]
    
    # Add model predictions
    for name, model in models.items():
        X[f"{name}_pred"] = model.predictions['train']['prediction']
    
    y = train_data['PremiumAmount']
    
    # Train meta-model
    ensemble = RandomForestRegressor()
    ensemble.fit(X, y)
    return ensemble

if __name__ == "__main__":
    train_scores_path = [f"ensemble/output/train_scores_{model}.csv" for model in ["SVR", "RandomForest", "LightGBM", "CatBoost", "KNN", "LinearRegression", "XGBoost", "NeuralNet"]]
    validation_scores_path = [f"ensemble/output/validation_scores_{model}.csv" for model in ["SVR", "RandomForest", "LightGBM", "CatBoost", "KNN", "LinearRegression", "XGBoost", "NeuralNet"]]

    ensemble_model = train_ensemble()