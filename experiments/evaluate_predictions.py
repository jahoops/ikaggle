import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
from experiments.data_manager import data_manager

target_column = "PremiumAmount"

# Load the best model and scalers
model_path = os.path.join("output", "Svr", "model", "best_model.joblib")
print("Model path:", model_path)
model = joblib.load(model_path)

model_dir = os.path.join("output", "Svr", "model")
scaler = joblib.load(os.path.join(model_dir, "robust_scaler.joblib"))
target_scaler = joblib.load(os.path.join(model_dir, "target_scaler.joblib"))
feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.joblib"))

# Load validation data
data = data_manager.finalvalidate.copy()
y_true_original = data[target_column].values

# Scale features and make predictions
X = data[feature_cols]
X_scaled = scaler.transform(X)
y_pred_scaled = model.predict(X_scaled)

# Print raw data types and shapes for debugging
print("\nData types and shapes:")
print(f"y_pred_scaled type: {type(y_pred_scaled)}")
print(f"y_pred_scaled shape: {y_pred_scaled.shape if hasattr(y_pred_scaled, 'shape') else 'no shape'}")

# Convert predictions to numpy array if needed
y_pred_scaled = np.array(y_pred_scaled)

# Scale true values for comparison
y_true_scaled = target_scaler.transform(y_true_original.reshape(-1, 1)).flatten()

# Create aligned results DataFrame with scaled values
results = pd.DataFrame({
    'id': data['id'],
    'true_value_scaled': y_true_scaled,
    'prediction_scaled': y_pred_scaled
})

print("\nVerifying ID alignment (scaled values):")
print(results.head())

# Print ranges at each step
print("\nPrediction ranges:")
print(f"Raw predictions (scaled): [{y_pred_scaled.min():.2f}, {y_pred_scaled.max():.2f}]")
print(f"True values (original): [{y_true_original.min():.2f}, {y_true_original.max():.2f}]")
print(f"True values (scaled): [{y_true_scaled.min():.2f}, {y_true_scaled.max():.2f}]")

# Add scaling verification
print("\nScaling Verification:")
print("1. Original ranges:")
print(f"X: [{X.min().min():.2f}, {X.max().max():.2f}]")
print(f"y: [{y_true_original.min():.2f}, {y_true_original.max():.2f}]")

print("\n2. After scaling:")
# Handle cuDF Series/DataFrame min/max
x_scaled_min = float(X_scaled.min().min() if hasattr(X_scaled, 'min') else X_scaled.min())
x_scaled_max = float(X_scaled.max().max() if hasattr(X_scaled, 'max') else X_scaled.max())
print(f"X_scaled: [{x_scaled_min:.2f}, {x_scaled_max:.2f}]")
print(f"y_true_scaled: [{y_true_scaled.min():.2f}, {y_true_scaled.max():.2f}]")
print(f"y_pred_scaled: [{y_pred_scaled.min():.2f}, {y_pred_scaled.max():.2f}]")

# Add more detailed scaling analysis
print("\n3. Scaling Statistics:")
print("Predictions:")
print(f"Mean: {np.mean(y_pred_scaled):.4f}")
print(f"Std: {np.std(y_pred_scaled):.4f}")
print("\nTrue Values:")
print(f"Mean: {np.mean(y_true_scaled):.4f}")
print(f"Std: {np.std(y_true_scaled):.4f}")

# Inverse transform predictions
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
print(f"Predictions (unscaled): [{y_pred.min():.2f}, {y_pred.max():.2f}]")

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'id': data['id'],
    'true_value': y_true_original,
    'predicted': y_pred,
    'abs_error': np.abs(y_pred - y_true_original)
})

print("\nFirst 5 rows comparison:")
print(comparison_df.head())

# Compute error statistics
errors = comparison_df['abs_error']
error_threshold = np.percentile(errors, 90)

print("\nError Analysis:")
print(f"Mean Error: {errors.mean():.2f}")
print(f"Median Error: {errors.median():.2f}")
print(f"90th percentile (threshold): {error_threshold:.2f}")

# Identify poorly estimated samples
poor_samples = comparison_df[comparison_df['abs_error'] >= error_threshold].copy()

# Check if any samples are above the threshold
if len(poor_samples) == 0:
    print("\nAll samples are well-estimated based on the current threshold.")
else:
    # Add features to analysis DataFrames
    poor_samples_with_features = poor_samples.merge(
        data[feature_cols], 
        left_index=True, 
        right_index=True
    )

    all_predictions_with_features = comparison_df.merge(
        data[feature_cols], 
        left_index=True, 
        right_index=True
    )

    # Save outputs
    output_dir = "experiments"
    os.makedirs(output_dir, exist_ok=True)

    poor_samples_with_features.to_csv(
        os.path.join(output_dir, "poorly_estimated_samples.csv"), 
        index=False
    )
    all_predictions_with_features.to_csv(
        os.path.join(output_dir, "all_predictions.csv"), 
        index=False
    )

    print(f"\nFound {len(poor_samples)} poorly estimated samples.")
    print(f"Results saved to:\n {output_dir}/poorly_estimated_samples.csv\n {output_dir}/all_predictions.csv")