import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_log_error
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(train):
    runBigNet(train)

def runBigNet(data):
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Num GPUs Available: {len(gpus)}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.warning("No GPUs available. Using CPU.")

    # Split data into features and target
    X = data.drop(columns=['PremiumAmount'])
    y = data['PremiumAmount']
    
    # Ensure target values are positive
    y_min = y.min()
    if y_min <= 0:
        y += abs(y_min) + 1
    
    # Check for string values in the dataset
    for col in X.columns:
        if X[col].dtype == 'object':
            logger.warning(f"Column {col} contains string values. Converting to numeric.")
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Define the neural network model
    def create_model():
        model = Sequential([
            tf.keras.Input(shape=(X.shape[1],)),  # Use Input layer
            Dense(64, activation='relu'),  # Reduced number of units
            Dropout(0.2),
            Dense(32, activation='relu'),  # Reduced number of units
            Dropout(0.2),
            Dense(16, activation='relu'),  # Reduced number of units
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_logarithmic_error')
        return model
    
    # Set up callbacks for checkpointing and early stopping
    checkpoint_path = 'ensembles/intermediate/bignet_model_checkpoint.h5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmsle_scores = []
    poorly_estimated_rows = []
    firstRun = True
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        model = create_model()
        
        # Check if a checkpoint exists and load weights
        if firstRun and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Checkpoint found at {checkpoint_path}. Resuming training...")
                model.load_weights(checkpoint_path)
                firstRun = False
            except ValueError as e:
                logger.warning(f"Checkpoint mismatch: {e}. Starting training from scratch...")
                os.remove(checkpoint_path)
        else:
            logger.info("No checkpoint found. Starting training from scratch...")
        
        # Train the model
        logger.info("Starting model training...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1, callbacks=[checkpoint, early_stopping])  # Reduced batch size
        
        # Load the best model
        model.load_weights(checkpoint_path)
        
        # Evaluate the model
        y_pred = model.predict(X_val)
        
        # Ensure predictions are positive
        y_pred = np.maximum(y_pred, 0)
        
        rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
        rmsle_scores.append(rmsle)
        logger.info(f"Fold Validation RMSLE: {rmsle:.4f}")
        
        # Identify poorly estimated rows
        abs_diff = np.abs(y_val - y_pred.flatten())
        threshold = np.percentile(abs_diff, 90)  # Top 10% worst predictions
        poorly_estimated = data.iloc[val_index][abs_diff > threshold]
        poorly_estimated_rows.append(poorly_estimated)
    
    # Calculate average RMSLE across folds
    avg_rmsle = np.mean(rmsle_scores)
    logger.info(f"Average Validation RMSLE: {avg_rmsle:.4f}")
    
    # Save the final model
    final_model_path = 'ensembles/intermediate/bignet_model.h5'
    model.save(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Save poorly estimated rows for further analysis
    poorly_estimated_df = pd.concat(poorly_estimated_rows)
    poorly_estimated_df.to_csv('ensembles/intermediate/poorly_estimated_rows.csv', index=False)
    logger.info("Saved poorly estimated rows to ensembles/intermediate/poorly_estimated_rows.csv")

    # Print the current score
    print(f"Average RMSLE score: {avg_rmsle:.4f}")

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    train = pd.read_csv(f"ensembles/intermediate/merged_train.csv")
    start(train)
