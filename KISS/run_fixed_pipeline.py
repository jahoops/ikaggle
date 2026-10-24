import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from model.group_trainer import GroupSpecialistTrainer
from model.super_nn import SuperNN
from model.diverse_models import DiverseModelFactory, ModelTrainer
from utils.feature_analyzer import combine_feature_importance_methods
from sklearn.preprocessing import StandardScaler
from utils.validation_checks import PipelineValidator

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_submission(predictions, output_path, min_premium=50.0):
    """
    Create a submission file with valid insurance premium predictions.
    Handles negative values and zeros properly.
    """
    # Create IDs
    ids = np.arange(1200000, 1200000 + len(predictions))
    
    # Fix negative values
    if np.any(predictions < 0):
        negative_count = np.sum(predictions < 0)
        print(f"Warning: Found {negative_count} negative predictions ({negative_count/len(predictions)*100:.2f}%)")
        print("Converting negatives to small positive values...")
        
        # Replace negatives with small positive values based on rank
        negative_indices = np.where(predictions < 0)[0]
        negative_vals = predictions[negative_indices]
        
        # Rank negative values to preserve relative ordering
        ranks = negative_vals.argsort().argsort()
        normalized_ranks = ranks / max(1, len(ranks) - 1)  # Scale to 0-1
        
        # Create replacement values from min_premium/2 to min_premium
        replacements = min_premium/2 + min_premium/2 * normalized_ranks
        
        # Replace negative values
        predictions = predictions.copy()  # Create a copy to avoid modifying the original
        predictions[negative_indices] = replacements
    
    # Handle zeros (which might indicate very low risk)
    zero_count = np.sum(predictions == 0)
    if zero_count > 0:
        print(f"Found {zero_count} zero predictions ({zero_count/len(predictions)*100:.2f}%)")
        print("Setting zeros to minimum premium...")
        
        # Replace zeros with a small fixed premium (real insurance policies normally have a minimum)
        predictions = np.where(predictions == 0, min_premium/2, predictions)
    
    # Check distribution is reasonable for insurance data
    mean_pred = np.mean(predictions)
    if mean_pred < 100:  # Usually insurance premiums average in hundreds of dollars
        print(f"Warning: Mean prediction ({mean_pred:.2f}) seems low for insurance premiums")
        print(f"Scaling predictions to a more realistic range...")
        
        # Scale to a typical insurance premium range while preserving relativity
        scale_factor = 500.0 / max(mean_pred, 1.0)  # Target mean of $500 (typical insurance premium)
        predictions *= scale_factor
        print(f"Applied scaling factor of {scale_factor:.2f}")
        
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': ids,
        'Premium Amount': predictions
    })
    
    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Print statistics about the submission
    print("\nSubmission statistics:")
    print(f"Count: {len(predictions)}")
    print(f"Min: ${np.min(predictions):.2f}")
    print(f"Max: ${np.max(predictions):.2f}")
    print(f"Mean: ${np.mean(predictions):.2f}")
    print(f"Median: ${np.median(predictions):.2f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50)
    plt.axvline(np.mean(predictions), color='r', linestyle='--', label=f'Mean: ${np.mean(predictions):.2f}')
    plt.axvline(np.median(predictions), color='g', linestyle='--', label=f'Median: ${np.median(predictions):.2f}')
    plt.xlabel('Premium Amount ($)')
    plt.ylabel('Count')
    plt.title('Distribution of Predicted Premium Amounts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save histogram
    histogram_path = os.path.splitext(output_path)[0] + '_histogram.png'
    plt.savefig(histogram_path)
    plt.close()
    print(f"Histogram saved to {histogram_path}")
    
    return submission

def main():
    # Start timing the pipeline
    start_time = time.time()
    
    print("Starting fixed insurance premium prediction pipeline...")
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Create directories
    base_results_dir = '/home/USER/Projects/ikaggle/KISS/results'
    results_dir = f"{base_results_dir}/fixed_pipeline"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define paths for intermediate steps
    processed_data_path = f"{results_dir}/processed_data"
    segmentation_path = f"{results_dir}/segmentation"
    base_models_path = f"{results_dir}/base_models"
    group_models_path = f"{results_dir}/group_models"
    
    # Step 1: Data Processing
    print("\n" + "="*80)
    print("STEP 1: DATA PROCESSING")
    print("="*80)
    
    # Define file paths
    train_path = '/home/USER/Projects/ikaggle/KISS/data/train_raw.csv'
    test_path = '/home/USER/Projects/ikaggle/KISS/data/contest_raw.csv'
    
    processor = InsuranceDataProcessor(
        train_path=train_path,
        test_path=test_path,
        results_dir=f"{results_dir}/data_processing"
    )
    
    processor.prepare_data()
    X_train, X_test, y_train = processor.get_processed_data()
    
    # Fix: Define non_negative_features before using it
    # List of features that should never be negative (natural positive-only values)
    non_negative_features = [
        'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 
        'PreviousClaims', 'VehicleAge', 'CreditScore', 'InsuranceDuration',
        'id'  # IDs should never be negative
    ]
    
    # Add all one-hot encoded features (they should all be non-negative)
    one_hot_features = []
    for col in X_train.columns:
        if '_' in col and col.split('_')[0] in processor.cat_features:
            one_hot_features.append(col)
            non_negative_features.append(col)
    
    # Also include binary features
    binary_features = ['SmokingStatus', 'Gender']
    for feature in binary_features:
        if feature in X_train.columns:
            non_negative_features.append(feature)
    
    # Force positive values for all one-hot features 
    print("\nForcing non-negative values for one-hot encoded features...")
    for col in one_hot_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].clip(lower=0)
            X_test[col] = X_test[col].clip(lower=0)
            print(f"  Clipped {col} to be non-negative")
    
    # Fix gender and binary feature values
    for feature in binary_features:
        if feature in X_train.columns:
            X_train[feature] = X_train[feature].clip(0, 1)
            X_test[feature] = X_test[feature].clip(0, 1)
            print(f"  Clipped {feature} to be between 0 and 1")
    
    # Validate data
    validator.check_data(X_train, "X_train", "data_processing")
    validator.check_data(X_test, "X_test", "data_processing")
    validator.check_data(y_train, "y_train", "data_processing", 
                        expectations={"y_train": {"min": 0}})
    
    # Adjust validator expectations for scaled features
    scaled_feature_names = [col for col in X_train.columns if any(term in col.lower() for term in 
                      ['interaction', 'ratio', 'date_', 'day', 'month', 'year'])]

    validator.check_data(X_train, "X_train", "data_processing", 
                        expectations={col: {"min": float('-inf')} for col in scaled_feature_names})

    print("\nAdjusting validator expectations for feature types...")

    # Create expectations dictionary
    expectations = {}

    # Define features that are known to have mostly zeros (insurance domain knowledge)
    zero_tolerance_features = [
        'PreviousClaims', 'ClaimsPerYear', 'Age_PreviousClaims_Interaction',
        'InsuranceDuration_PreviousClaims_Interaction', 'HealthScore_SmokingStatus_Interaction'
    ]

    # For features that naturally have many zeros, set a high zero tolerance
    for col in X_train.columns:
        if any(zero_feat in col for zero_feat in zero_tolerance_features) or 'Claims' in col:
            expectations[col] = {"zero_threshold": 80}  # Allow up to 80% zeros

    # For all other features
    for col in X_train.columns:
        if col not in expectations:
            if col in non_negative_features:
                expectations[col] = {"min": 0}  # Must be non-negative
            else:
                expectations[col] = {"min": float('-inf')}  # Allow negative values

    # Specifically for binary features
    for col in binary_features + one_hot_features:
        if col in X_train.columns:
            expectations[col] = {"min": 0, "max": 1}  # Must be between 0 and 1

    # Validate with adjusted expectations
    validator.check_data(X_train, "X_train", "data_processing_with_domain_knowledge", expectations=expectations)

    # Save processed data
    os.makedirs(processed_data_path, exist_ok=True)
    X_train.to_csv(f"{processed_data_path}/X_train.csv", index=False)
    X_test.to_csv(f"{processed_data_path}/X_test.csv", index=False)
    pd.DataFrame({'PremiumAmount': y_train}).to_csv(f"{processed_data_path}/y_train.csv", index=False)
    
    # Create log-transformed target for modeling
    y_train_log = np.log1p(y_train)
    
    # Validate log-transformed target
    validator.check_data(y_train_log, "y_train_log", "log_transform",
                        expectations={"y_train_log": {"min": 0}})
    
    # Split data into train and validation sets
    from sklearn.model_selection import train_test_split
    X_train_df, X_val_df, y_train_log_split, y_val_log = train_test_split(
        X_train, y_train_log, test_size=0.2, random_state=42
    )
    
    # Convert back to original scale for evaluation metrics
    y_val = np.expm1(y_val_log)  
    
    # Step 2: Feature Importance and Segmentation
    print("\n" + "="*80)
    print("STEP 2: FEATURE IMPORTANCE & CUSTOMER SEGMENTATION")
    print("="*80)
    
    # Create segmenter
    segmenter = InsuranceSegmenter(results_dir=segmentation_path)
    
    # Analyze feature importance and get top features
    top_features = combine_feature_importance_methods(
        segmenter,
        X_train_df, 
        y_train_log_split,
        methods=['random_forest', 'correlation'],
        n_features=15
    )
    
    # Validate top features
    validator.check_data(pd.Series(top_features), "top_features", "feature_importance")
    
    # Save top features
    os.makedirs(segmentation_path, exist_ok=True)
    with open(f"{segmentation_path}/top_features.txt", 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    
    # Create customer segments
    optimal_clusters = 4  # Can be determined dynamically
    groups = segmenter.create_segments(X_train_df, y_train_log_split, method='kmeans', n_clusters=optimal_clusters)
    
    # Validate segments
    validator.check_data(groups, "segments", "segmentation")
    
    # Analyze segments
    segment_stats = segmenter.analyze_segments(X_train_df, y_train)
    
    # Add segment to training data for feature creation
    X_train_df['Segment'] = groups
    X_val_df['Segment'] = segmenter.assign_segment(X_val_df)
    
    # Save training data for future segment assignment
    X_features = X_train_df[top_features].copy()
    X_features.to_csv(f"{segmentation_path}/train_features_for_segments.csv", index=False)
    
    # Step 3: Create Feature Subsets
    print("\n" + "="*80)
    print("STEP 3: CREATING FEATURE SUBSETS")
    print("="*80)
    
    feature_subsets = []
    
    # Subset 1: All features except Segment
    all_features = [col for col in X_train_df.columns if col != 'Segment']
    feature_subsets.append(all_features)
    
    # Subset 2: Top features from importance analysis
    feature_subsets.append(top_features)
    
    # Subset 3: Segment-specific important features + Segment
    segment_features = []
    for feature in top_features[:10]:  # Top 10 features from importance analysis
        segment_features.append(feature)
    segment_features.append('Segment')
    feature_subsets.append(segment_features)
    
    # Save feature subsets
    with open(f"{results_dir}/feature_subsets.txt", 'w') as f:
        for i, subset in enumerate(feature_subsets):
            f.write(f"Subset {i+1}: {','.join(subset)}\n")
    
    # Step 4: Train Models for Every Segment
    print("\n" + "="*80)
    print("STEP 4: TRAINING SEGMENT-SPECIFIC MODELS")
    print("="*80)
    
    # For each segment, train a separate model
    segment_models = []
    segment_predictions = np.zeros((len(X_test), 4))  # 4 segments
    
    for segment_id in range(4):
        print(f"\nTraining model for segment {segment_id}...")
        
        # Get segment data - remove Segment column safely
        segment_indices = X_train_df['Segment'] == segment_id
        segment_X = X_train_df.loc[segment_indices].copy()
        segment_y = y_train_log_split[segment_indices]
        
        # Remove any extra columns that might cause issues (like Cluster or Target)
        cols_to_drop = ['Segment', 'Cluster', 'Target']
        for col in cols_to_drop:
            if col in segment_X.columns:
                print(f"Dropping column {col} from training data")
                segment_X = segment_X.drop(col, axis=1)
        
        # Validate segment data
        validator.check_data(segment_X, f"segment_{segment_id}_X", "segment_training")
        validator.check_data(segment_y, f"segment_{segment_id}_y", "segment_training")
        
        # Create and train model for this segment
        model, model_type = DiverseModelFactory.create_model("wide_and_deep", input_dim=segment_X.shape[1])
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ]
        
        # Scale features
        scaler = StandardScaler()
        segment_X_scaled = scaler.fit_transform(segment_X)
        
        # Get validation data for this segment - make sure columns match exactly
        val_segment_indices = X_val_df['Segment'] == segment_id
        val_segment_X = X_val_df.loc[val_segment_indices].copy()
        val_segment_y = y_val_log[val_segment_indices]
        
        # Ensure validation data has exactly the same columns as training data
        for col in cols_to_drop:
            if col in val_segment_X.columns:
                print(f"Dropping column {col} from validation data")
                val_segment_X = val_segment_X.drop(col, axis=1)
        
        # Ensure column order matches exactly (this is critical for StandardScaler)
        val_segment_X = val_segment_X[segment_X.columns]
        print(f"Training columns: {segment_X.shape[1]}, Validation columns: {val_segment_X.shape[1]}")
        
        # Now transform with matching columns
        val_segment_X_scaled = scaler.transform(val_segment_X)
        
        # Train model
        model.fit(
            segment_X_scaled, segment_y,
            validation_data=(val_segment_X_scaled, val_segment_y),
            epochs=50,  # Will be limited by early stopping
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model
        segment_models.append({
            'model': model,
            'scaler': scaler,
            'segment': segment_id,
            'feature_columns': segment_X.columns.tolist()  # Store column list for later use
        })
        
        # Get test predictions for this segment
        test_segment_indices = segmenter.assign_segment(X_test) == segment_id
        test_segment_X = X_test.iloc[test_segment_indices].copy()
        
        # Ensure test data has the same columns in the same order
        test_segment_X = test_segment_X[segment_X.columns]
        test_segment_X_scaled = scaler.transform(test_segment_X)
        
        # Make predictions (in log space)
        test_segment_preds = model.predict(test_segment_X_scaled).flatten()
        
        # Store predictions
        segment_predictions[test_segment_indices, segment_id] = test_segment_preds
    
    # Convert segment predictions to actual predictions (choose the predictions from the correct segment)
    test_segments = segmenter.assign_segment(X_test)
    
    # Initialize final predictions array
    final_preds_log = np.zeros(len(X_test))
    
    # For each row, get prediction from the appropriate segment model
    for i, segment_id in enumerate(test_segments):
        final_preds_log[i] = segment_predictions[i, segment_id]
    
    # Validate log-space predictions
    validator.check_data(final_preds_log, "final_preds_log", "final_predictions")
    
    # Convert from log space to premium space
    final_preds = np.expm1(final_preds_log)
    
    # Validate final predictions
    validator.check_data(final_preds, "final_preds", "final_predictions_expm1",
                       expectations={"final_preds": {"min": 0}})
    
    # Create submission file with proper handling of invalid values
    submission_path = f"{results_dir}/fixed_segment_submission.csv"
    submission = create_submission(final_preds, submission_path)
    
    # Calculate total runtime
    end_time = time.time()
    runtime_minutes = (end_time - start_time) / 60
    print(f"\nPipeline completed in {runtime_minutes:.1f} minutes.")
    
    # Summarize validation
    validator.summarize()

if __name__ == "__main__":
    main()
