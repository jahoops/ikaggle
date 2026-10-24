import sys
from pathlib import Path
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_log_error

# Ensure experiments module is in path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from experiments.data_manager import data_manager  # Import data_manager

# Load data
data = data_manager.finalvalidate.copy()

# Check if data is loaded correctly
if data is None:
    print("Error: Data not loaded. Check data_manager.py.")
    exit()

# Define high income threshold
high_income_threshold = 70000

# Split data into high income and low income
high_income_data = data[data['AnnualIncome'] >= high_income_threshold].copy()
low_income_data = data[data['AnnualIncome'] < high_income_threshold].copy()

# Define features and target
features = ['AnnualIncome', 'HealthScore', 'CreditScore', 'Age', 'NumberofDependents', 'SignupYear', 'InsuranceDuration', 'Location_Rural', 'VehicleAge', 'PreviousClaims']
target = 'PremiumAmount'

# Create interaction features
new_features = {}
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        new_features[features[i] + '*' + features[j]] = high_income_data[features[i]] * high_income_data[features[j]]

# Convert the dictionary to a DataFrame
new_features_df = pd.DataFrame(new_features)

# Concatenate the new features to the original DataFrame
high_income_data = pd.concat([high_income_data, new_features_df], axis=1)

# Update features list to include interaction terms
features = features + list(new_features.keys())

# Prepare data for training high income model
X_high = high_income_data[features]
y_high = high_income_data[target]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_high, y_high, test_size=0.2, random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, scoring='neg_mean_squared_log_error', cv=3, verbose=1, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the model using RMSLE
y_pred = best_model.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
print(f"RMSLE on high income data: {rmsle}")

# Save the model
joblib.dump(best_model, 'high_income_model.joblib')