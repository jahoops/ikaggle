import sys
from pathlib import Path
import os

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from utils import transform_data, data_manager
import pandas as pd
from sklearn.metrics import mean_squared_error

def load_predictions(data, match, train_data):
    """Load all CSV prediction files from input directory"""
    predictions = {}

    # create path to /ensemble/input directory
    input_dir = Path('/ensembles/input')

    for csv_file in input_dir.glob(match):
        # Extract model name and score from filename
        parts = csv_file.stem.split('_')
        model_name = parts[0]
        # Load predictions
        df = pd.read_csv(csv_file)
        # for each model, add two columns to the data train_data with the same ID: one for the prediction, and one for the accuracy as calculated against the original Premium Amount
        # calculate accuracy

        predictions[model_name] = {
            'id': df['id'],
            'prediction': df['Premium Amount'],
            'accuracy': df['Premium Amount'] / train_data['Premium Amount']
        }
        data = pd.merge(data, predictions, on='id', how='left')
    return train_data

# Load train data
train_data = transform_data(data_manager.train, data_manager.nn_train)

# Load all predictions
train_data = load_predictions(train_data, '*train_estimates_*.csv', train_data)
print(train_data.head())
print(train_data.columns)

# X_train: Features from original data + model outputs (estimated amounts)
X_train = train_data.drop(['id', 'PremiumAmount'], axis=1)

# y_train: Target variable (AMOUNT)
y_train = train_data['PremiumAmount']

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1))  # Single output for AMOUNT

model.compile(optimizer='adam', loss='mse')

# Create output directory
output_dir = Path('ensembles/output')
output_dir.mkdir(parents=True, exist_ok=True)


# Load test data
val_data = transform_data(data_manager.validate, data_manager.nn_validate)
test_data = transform_data(data_manager.test, data_manager.nn_test)
# X_train: Features from original data + model outputs (estimated amounts)
# Align columns:
X_val = val_data[train_data.columns].drop(['id'], axis=1)  # Don't drop PremiumAmount if not present
X_test = test_data[train_data.columns].drop(['id'], axis=1)  # Don't drop PremiumAmount if not present

# y_train: Target variable (AMOUNT)
y_val = test_data['PremiumAmount']
y_test = test_data['PremiumAmount']

# Debug data shapes
print(f"Train data shape: {train_data.shape}")
print(f"Validate data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")


# Split training data for validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop(columns=['id', 'PremiumAmount']),
    train_data['PremiumAmount'],
    test_size=0.2,
    random_state=42
)

# Train
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluate
vpredictions = model.predict(X_val).ravel()
score = mean_squared_error(y_val, vpredictions)
print(f"Validation MSE: {score}")

# Load validate predictions
val_data = load_predictions(val_data, '*test_estimates_*.csv', train_data)
print(val_data.head())
print(val_data.columns)

# Load test predictions
test_data = load_predictions(test_data, '*test_estimates_*.csv', train_data)
print(test_data.head())
print(test_data.columns)

# Drop both 'id' and 'PremiumAmount' from train_data columns before aligning
common_cols = train_data.drop(columns=['id', 'PremiumAmount']).columns

# score using validation data
# X_test: Features from original data + model outputs (estimated amounts)
X_val = val_data[common_cols]
X_test = test_data[common_cols]

# Train on full training set
X_train = train_data.drop(columns=['id', 'PremiumAmount'])
y_train = train_data['PremiumAmount']
model.fit(X_train, y_train, epochs=1, batch_size=32)

# Evaluate on validation set
X_val = val_data[common_cols]
y_val = val_data['PremiumAmount']
vpredictions = model.predict(X_val).ravel()
score = mean_squared_error(y_val, vpredictions)
print(f'Validation MSE: {score}')

# Predict on test set
X_test = test_data[common_cols]
tpredictions = model.predict(X_test).ravel()

# Save predictions
submission = pd.DataFrame({
    'id': test_data['id'],
    'Premium Amount': tpredictions.round(3)
})
submission_path = output_dir / 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f'Submission saved to {submission_path}')

