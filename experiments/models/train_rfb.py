# /mnt/c/home/USER/ensemble/train_rfb.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from utils.memory_monitor import setup_memory_logging, log_memory_usage

setup_memory_logging()

def train():
    log_memory_usage()  # Log before loading data
    
    # Load data in chunks
    df = pd.read_csv('data/processed/final_train.csv', chunksize=10000)
    X = pd.DataFrame()
    y = pd.Series()

    for chunk in df:
        X = pd.concat([X, chunk.drop('PremiumAmount', axis=1)])
        y = pd.concat([y, chunk['PremiumAmount']])

    # Free memory
    del df

    log_memory_usage()  # Log after loading data
    
    # Split with smaller test size
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # Free memory
    del X, y

    # Use fewer trees and limit depth
    rf = RandomForestRegressor(
        n_estimators=100,  # Reduced from default
        max_depth=10,      # Limit tree depth
        n_jobs=-1,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(rf, 'models/rf_basic.joblib')

train()