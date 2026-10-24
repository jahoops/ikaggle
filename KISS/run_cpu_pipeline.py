# CPU-compatible version of the pipeline that avoids GPU library errors

import os
# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import the necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from sklearn.preprocessing import StandardScaler
from utils.validation_checks import PipelineValidator
from sklearn.ensemble import GradientBoostingRegressor

# Print TensorFlow configuration to confirm CPU usage
print("TensorFlow is using CPU:", tf.config.list_physical_devices('CPU'))
print("TensorFlow version:", tf.__version__)

def create_submission(predictions, output_path, min_premium=50.0):
    """Create a submission file with valid insurance premium predictions."""
    # Same implementation as in run_fixed_pipeline.py
    # ...

def main():
    # Start timing the pipeline
    start_time = time.time()
    
    print("Starting CPU-compatible insurance premium prediction pipeline...")
    
    # Initialize validator and directories
    validator = PipelineValidator()
    # (Same as in run_fixed_pipeline.py)
    # ...
    
    # Data processing steps
    # (Same as in run_fixed_pipeline.py)
    # ...
    
    # Load and preprocess training data
    data_processor = InsuranceDataProcessor()
    X_train_df, y_train_log_split = data_processor.load_and_preprocess_training_data()
    X_val_df, y_val_log = data_processor.load_and_preprocess_validation_data()
    X_test = data_processor.load_and_preprocess_test_data()
    
    # Step 4: Train Models for Every Segment - MODIFIED TO USE SKLEARN MODELS
    print("\n" + "="*80)
    print("STEP 4: TRAINING SEGMENT-SPECIFIC MODELS (CPU-COMPATIBLE)")
    print("="*80)
    
    # For each segment, train a separate model using scikit-learn instead of TensorFlow
    segment_models = []
    segment_predictions = np.zeros((len(X_test), 4))  # 4 segments
    
    # Initialize the segmenter
    segmenter = InsuranceSegmenter()
    
    for segment_id in range(4):
        print(f"\nTraining model for segment {segment_id}...")
        
        # Get segment data - remove Segment column safely
        segment_indices = X_train_df['Segment'] == segment_id
        segment_X = X_train_df.loc[segment_indices].copy()
        segment_y = y_train_log_split[segment_indices]
        
        # Remove any extra columns that might cause issues
        cols_to_drop = ['Segment', 'Cluster', 'Target']
        for col in cols_to_drop:
            if col in segment_X.columns:
                print(f"Dropping column {col} from training data")
                segment_X = segment_X.drop(col, axis=1)
        
        # Validate segment data
        validator.check_data(segment_X, f"segment_{segment_id}_X", "segment_training")
        validator.check_data(segment_y, f"segment_{segment_id}_y", "segment_training")
        
        # Create and train CPU-friendly model (GBM instead of neural network)
        print(f"Training Gradient Boosting Regressor for segment {segment_id}...")
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            learning_rate=0.1,
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        segment_X_scaled = scaler.fit_transform(segment_X)
        
        # Get validation data for this segment
        val_segment_indices = X_val_df['Segment'] == segment_id
        val_segment_X = X_val_df.loc[val_segment_indices].copy()
        val_segment_y = y_val_log[val_segment_indices]
        
        # Ensure validation data has exactly the same columns
        for col in cols_to_drop:
            if col in val_segment_X.columns:
                val_segment_X = val_segment_X.drop(col, axis=1)
        
        val_segment_X = val_segment_X[segment_X.columns]
        val_segment_X_scaled = scaler.transform(val_segment_X)
        
        # Train model (scikit-learn model instead of keras)
        model.fit(segment_X_scaled, segment_y)
        
        # Evaluate on validation data
        val_score = np.sqrt(np.mean((model.predict(val_segment_X_scaled) - val_segment_y) ** 2))
        print(f"Validation RMSE for segment {segment_id}: {val_score:.4f}")
        
        # Store model
        segment_models.append({
            'model': model,
            'scaler': scaler,
            'segment': segment_id,
            'feature_columns': segment_X.columns.tolist()
        })
        
        # Get test predictions for this segment
        test_segment_indices = segmenter.assign_segment(X_test) == segment_id
        test_segment_X = X_test.iloc[test_segment_indices].copy()
        test_segment_X = test_segment_X[segment_X.columns]
        test_segment_X_scaled = scaler.transform(test_segment_X)
        
        # Make predictions (in log space)
        test_segment_preds = model.predict(test_segment_X_scaled)
        
        # Store predictions
        segment_predictions[test_segment_indices, segment_id] = test_segment_preds
    
    # Rest of the pipeline (same as run_fixed_pipeline.py)
    # ...

if __name__ == "__main__":
    main()
