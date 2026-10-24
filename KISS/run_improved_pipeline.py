"""
Improved pipeline with error segment approach - modifications after initial test.
With robust checkpoint handling and debugging.
"""
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import glob
import argparse
import pickle
import joblib
from utils.data_processor import InsuranceDataProcessor
from utils.segmentation import InsuranceSegmenter
from utils.error_analysis import ErrorAnalyzer, EnhancedSegmenter
from utils.validation_checks import PipelineValidator
from utils.model_builder import build_insurance_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.model_fixes import diagnose_predictions, fix_model_initialization
from utils.prediction_fixes import reverse_scaling, analyze_predictions, extract_scaling_info_from_processor

# Define utility functions at module level (outside of any function)
def safe_predict(model, X, segment_id=None, batch_size=32, verbose=0):
    """
    Make predictions with additional safeguards against TF numerical issues
    """
    try:
        # First try normal prediction
        predictions = model.predict(X, batch_size=batch_size, verbose=verbose)
        
        # Check for NaNs or infinities
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print(f"Warning: Found NaN/Inf predictions in segment {segment_id}. Applying fallback.")
            # Switch to CPU for this prediction
            with tf.device('/CPU:0'):
                predictions = model.predict(X, batch_size=batch_size, verbose=verbose)
                
        # Final cleanup of any remaining NaNs or infinities
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=100.0, neginf=0.0)
        return predictions.flatten()
        
    except Exception as e:
        print(f"Error in prediction for segment {segment_id}: {str(e)}")
        # Return zeros as fallback
        return np.zeros(len(X))

def ensemble_predictions(models, X, segment_id=None, weights=None):
    """
    Create an ensemble of predictions from multiple models for more robustness
    """
    if not models:
        return np.zeros(len(X))
    
    all_preds = []
    for i, model in enumerate(models):
        try:
            preds = safe_predict(model, X, segment_id=f"{segment_id}-{i}")
            all_preds.append(preds)
        except Exception as e:
            print(f"Error in model {i} for segment {segment_id}: {str(e)}")
    
    if not all_preds:
        return np.zeros(len(X))
    
    # Use weights if provided, otherwise equal weighting
    if weights and len(weights) == len(all_preds):
        weighted_preds = np.zeros_like(all_preds[0])
        for i, preds in enumerate(all_preds):
            weighted_preds += weights[i] * preds
        return weighted_preds
    else:
        # Simple average
        return np.mean(all_preds, axis=0)

def calibrate_predictions(predictions, segment_id=None, calibration_factor=0.9):
    """
    Apply calibration to predictions based on segment
    
    This can help correct systematic biases in model predictions,
    which is especially important for error-prone segments.
    """
    # Default calibration - mild shrinkage toward the mean for stability
    adjusted_preds = predictions * calibration_factor
    
    # Special handling for error segment (assuming it's the last segment)
    if segment_id is not None and segment_id == 4:  # Error segment
        # More aggressive shrinkage for error segment
        adjusted_preds = predictions * (calibration_factor * 0.95)
        # Add a small constant to avoid extreme low values
        adjusted_preds += 0.05
        
    return adjusted_preds

def weighted_segment_blending(segment_predictions, segment_weights=None):
    """
    Create a weighted blend of segment predictions
    
    This can help when some segments are more reliable than others.
    """
    n_segments = len(segment_predictions)
    n_samples = len(segment_predictions[0])
    
    # Default to equal weights if none provided
    if segment_weights is None:
        segment_weights = [1.0 / n_segments] * n_segments
        # Downweight error segment (assuming it's the last one)
        if n_segments >= 4:  # If we have enough segments
            segment_weights[-1] *= 0.7  # Reduce weight of error segment
            # Normalize weights
            total = sum(segment_weights)
            segment_weights = [w / total for w in segment_weights]
    
    # Blend predictions
    blended = np.zeros(n_samples)
    for i in range(n_segments):
        blended += segment_weights[i] * segment_predictions[i]
        
    return blended

# Copy create_submission function from run_enhanced_pipeline.py
def create_submission(predictions, output_path, min_premium=50.0):
    # Before applying min_premium, diagnose raw predictions
    diagnose_predictions(predictions, title="Raw Predictions")
    
    # Apply a dynamic minimum premium threshold based on distribution
    if np.median(predictions) < 100:  # If predictions are generally too low
        print("WARNING: Predictions appear to be too low, using a lower minimum threshold")
        min_premium = max(20.0, np.percentile(predictions, 5))  # Use 5th percentile or 20.0
    
    # Ensure predictions are positive
    predictions = np.maximum(predictions, 0.0)
    
    # Apply minimum premium threshold
    capped_predictions = np.maximum(predictions, min_premium)
    
    # Diagnose final predictions
    diagnose_predictions(capped_predictions, predictions, title="Final Predictions")
    
    # Create submission data
    submission_df = pd.DataFrame({
        'id': range(1200000, 1200000 + len(capped_predictions)),
        'Premium Amount': capped_predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    
    # Print distribution information
    print(f"Final prediction stats: min={np.min(capped_predictions):.2f}, "
          f"max={np.max(capped_predictions):.2f}, "
          f"mean={np.mean(capped_predictions):.2f}, "
          f"median={np.median(capped_predictions):.2f}")
    
    return submission_df

def check_checkpoint_exists(filepath, description, skip_flag):
    """Check if a checkpoint exists and print appropriate message"""
    if skip_flag and os.path.exists(filepath):
        print(f"✓ {description} checkpoint exists, will be loaded")
        return True
    elif skip_flag and not os.path.exists(filepath):
        print(f"⚠ {description} checkpoint NOT found, will run this step")
        return False
    return skip_flag

def main():
    # Setup argument parser to control which steps to run
    parser = argparse.ArgumentParser(description='Run improved pipeline with error segment')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-base-segmentation', action='store_true', help='Skip base segmentation')
    parser.add_argument('--skip-error-analysis', action='store_true', help='Skip error analysis')
    parser.add_argument('--skip-model-training', action='store_true', help='Skip model training')
    parser.add_argument('--only-predictions', action='store_true', help='Only make predictions using existing models')
    args = parser.parse_args()
    
    # Start timing the pipeline
    start_time = time.time()
    
    print("\n" + "="*80)
    print("PIPELINE: IMPROVED INSURANCE PREMIUM PREDICTION WITH ERROR SEGMENT")
    print("="*80)
    
    # Setup directories
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    results_dir = os.path.join(os.path.dirname(__file__), "results", "improved_pipeline")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define checkpoint paths
    data_checkpoint = os.path.join(checkpoint_dir, "processed_data.pkl")
    base_segment_checkpoint = os.path.join(checkpoint_dir, "base_segments.pkl")
    enhanced_segment_checkpoint = os.path.join(checkpoint_dir, "enhanced_segments.pkl")
    models_checkpoint = os.path.join(checkpoint_dir, "segment_models.pkl")
    
    # Print configuration summary
    print("\nCONFIGURATION:")
    print(f"Data directory: {DATA_DIR}")
    print(f"Results directory: {results_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("\nRUNNING WITH FLAGS:")
    print(f"Skip preprocessing: {args.skip_preprocessing}")
    print(f"Skip base segmentation: {args.skip_base_segmentation}")
    print(f"Skip error analysis: {args.skip_error_analysis}")
    print(f"Skip model training: {args.skip_model_training}")
    print(f"Only predictions: {args.only_predictions}")
    
    # Check if checkpoint files exist
    print("\nCHECKING CHECKPOINTS:")
    can_skip_preprocessing = check_checkpoint_exists(data_checkpoint, "Data processing", 
                                                  args.skip_preprocessing or args.only_predictions)
    can_skip_base_segmentation = check_checkpoint_exists(base_segment_checkpoint, "Base segmentation", 
                                                      args.skip_base_segmentation or args.only_predictions)
    can_skip_error_analysis = check_checkpoint_exists(enhanced_segment_checkpoint, "Error analysis", 
                                                   args.skip_error_analysis or args.only_predictions)
    can_skip_model_training = check_checkpoint_exists(models_checkpoint, "Model training", 
                                                   args.skip_model_training or args.only_predictions)
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Step 1: Data Loading and Processing
    if can_skip_preprocessing:
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA FROM CHECKPOINT")
        print("="*80)
        
        try:
            with open(data_checkpoint, 'rb') as f:
                data = pickle.load(f)
                X_train_df = data['X_train']
                X_val_df = data['X_val'] 
                X_test_df = data['X_test']
                y_train = data['y_train']
                y_val = data['y_val']
                y_train_log_split = data['y_train_log']
                y_val_log = data['y_val_log']
                
            print(f"Successfully loaded data from checkpoint")
            print(f"Train shape: {X_train_df.shape}, Validation shape: {X_val_df.shape}, Test shape: {X_test_df.shape}")
        except Exception as e:
            print(f"ERROR loading data checkpoint: {str(e)}")
            print("Falling back to full preprocessing")
            can_skip_preprocessing = False
    
    if not can_skip_preprocessing:
        print("\n" + "="*80)
        print("STEP 1: LOADING AND PROCESSING DATA")
        print("="*80)
        
        # Define file paths
        train_path = os.path.join(DATA_DIR, "train_raw.csv")
        test_path = os.path.join(DATA_DIR, "contest_raw.csv")
        
        # Check if files exist
        print(f"Checking data files:")
        print(f"Train file exists: {os.path.exists(train_path)}")
        print(f"Test file exists: {os.path.exists(test_path)}")
        
        # Initialize data processor with required paths
        processor = InsuranceDataProcessor(train_path, test_path)
        
        # Load data
        print("Loading data...")
        processor.load_data()

        # Run the full data preparation pipeline
        print("Preparing data (cleaning, feature engineering, encoding)...")
        processor.prepare_data()  # This calls all the necessary methods
        
        # Get processed data
        X_train_processed, X_test_processed, y_train_processed = processor.get_processed_data()
        
        # Create test dataframe
        X_test_df = X_test_processed.copy()
        
        # Create train/validation split manually since there's no split_data method
        print("Splitting data into train/validation...")
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_train_processed, y_train_processed, 
            test_size=0.2, random_state=42
        )
        
        # Apply log transformation for regression task
        y_train_log_split = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        # Check data quality
        validator.check_data_quality(X_train_df, y_train, X_test_processed)
        
        # Save checkpoint
        with open(data_checkpoint, 'wb') as f:
            pickle.dump({
                'X_train': X_train_df,
                'X_val': X_val_df,
                'X_test': X_test_df,
                'y_train': y_train,
                'y_val': y_val,
                'y_train_log': y_train_log_split,
                'y_val_log': y_val_log
            }, f)
        print(f"Data checkpoint saved to {data_checkpoint}")

        # Extract and save scaling information for later use
        print("Extracting and saving scaling information for later use...")
        scaling_info = extract_scaling_info_from_processor(processor)
        
        # Save scaling info to checkpoint directory
        with open(os.path.join(checkpoint_dir, "target_scaling_info.pkl"), 'wb') as f:
            pickle.dump(scaling_info, f)
        print("Scaling information saved")

    # Step 2-3: Segmentation
    if can_skip_error_analysis and can_skip_base_segmentation:
        print("\n" + "="*80)
        print("STEP 2-3: LOADING ENHANCED SEGMENTS FROM CHECKPOINT")
        print("="*80)
        
        try:
            with open(enhanced_segment_checkpoint, 'rb') as f:
                segments_data = pickle.load(f)
                enhanced_segmenter = segments_data['segmenter']
                X_train_df['Segment'] = segments_data['train_segments']
                X_val_df['Segment'] = segments_data['val_segments'] 
                X_test_df['Segment'] = segments_data['test_segments']
                
            print(f"Successfully loaded enhanced segments from checkpoint")
            segment_counts = X_train_df['Segment'].value_counts().sort_index()
            print(f"Segment distribution:")
            for segment_id, count in segment_counts.items():
                print(f"  Segment {segment_id}: {count} samples ({count/len(X_train_df)*100:.1f}%)")
        except Exception as e:
            print(f"ERROR loading enhanced segments: {str(e)}")
            print("Falling back to creating segments")
            can_skip_error_analysis = False
            can_skip_base_segmentation = False
    
    # Handle base segmentation
    if not can_skip_error_analysis and can_skip_base_segmentation:
        print("\n" + "="*80)
        print("STEP 2: LOADING BASE SEGMENTS FROM CHECKPOINT")
        print("="*80)
        
        try:
            with open(base_segment_checkpoint, 'rb') as f:
                segment_data = pickle.load(f)
                segmenter = segment_data['segmenter']
                base_segments = segment_data['segments']
                
            print(f"Successfully loaded base segments from checkpoint")
            
            # Add segmentation to data
            X_train_df['Segment'] = base_segments
            X_train_df['Cluster'] = base_segments  # Keep original clusters
            
            # Assign segments to validation
            X_val_df['Segment'] = segmenter.assign_segment(X_val_df)
            X_val_df['Cluster'] = X_val_df['Segment']  # Keep original clusters
        except Exception as e:
            print(f"ERROR loading base segments: {str(e)}")
            print("Falling back to creating base segments")
            can_skip_base_segmentation = False
    
    if not can_skip_base_segmentation:
        print("\n" + "="*80)
        print("STEP 2: CREATING BASE SEGMENTS")
        print("="*80)
        
        # Create segmentation
        print("Creating customer segments...")
        segmenter = InsuranceSegmenter()
        base_segments = segmenter.create_segments(X_train_df, y_train_log_split, method='kmeans', n_clusters=4)
        
        # Save checkpoint
        with open(base_segment_checkpoint, 'wb') as f:
            pickle.dump({
                'segmenter': segmenter,
                'segments': base_segments
            }, f)
        print(f"Base segmentation checkpoint saved to {base_segment_checkpoint}")
    
        # Add segmentation to data
        X_train_df['Segment'] = base_segments
        X_train_df['Cluster'] = base_segments  # Keep original clusters
        
        # Assign segments to validation
        X_val_df['Segment'] = segmenter.assign_segment(X_val_df)
        X_val_df['Cluster'] = X_val_df['Segment']  # Keep original clusters
    
    # Step 3: Error Analysis & Enhanced Segmentation
    if not can_skip_error_analysis:
        print("\n" + "="*80)
        print("STEP 3: CREATING ENHANCED SEGMENTATION")
        print("="*80)
        
        # Create the base model to get validation predictions
        X_train_temp = X_train_df.drop(['Segment', 'Cluster'], axis=1, errors='ignore')
        X_val_temp = X_val_df.drop(['Segment', 'Cluster'], axis=1, errors='ignore')
        
        # Train a simple model to identify poorly predicted samples
        print("Training a simple model to identify poorly predicted samples...")
        model_temp = build_insurance_model(X_train_temp.shape[1], 'normal')
        model_temp.fit(X_train_temp, y_train_log_split, 
                      epochs=10, batch_size=32, verbose=0,
                      validation_data=(X_val_temp, y_val_log))
        
        # Get validation predictions
        val_preds = model_temp.predict(X_val_temp, verbose=0).flatten()
        
        # Enhanced segmentation with error analysis - MODIFIED ERROR THRESHOLD
        print("Creating enhanced segments with reduced error threshold...")
        error_analyzer = ErrorAnalyzer(error_threshold=0.92)  # Top 8% of errors (reduced from 15%)
        enhanced_segmenter = EnhancedSegmenter(base_segments=4, error_analyzer=error_analyzer)
        
        # Create enhanced segments
        segments = enhanced_segmenter.create_enhanced_segments(
            X_train_df, y_train_log_split,
            X_val=X_val_df, y_val=y_val_log, y_val_pred=val_preds
        )
        
        # Add segmentation to data
        X_train_df['Segment'] = segments
        
        # Assign segments to validation & test
        val_segments = enhanced_segmenter.assign_enhanced_segment(X_val_df)
        X_val_df['Segment'] = val_segments
        test_segments = enhanced_segmenter.assign_enhanced_segment(X_test_df)
        X_test_df['Segment'] = test_segments
        
        # Save checkpoint
        with open(enhanced_segment_checkpoint, 'wb') as f:
            pickle.dump({
                'segmenter': enhanced_segmenter,
                'train_segments': segments,
                'val_segments': val_segments,
                'test_segments': test_segments
            }, f)
        print(f"Enhanced segmentation checkpoint saved to {enhanced_segment_checkpoint}")
    
    # Step 4: Model Training
    if can_skip_model_training:
        print("\n" + "="*80)
        print("STEP 4: LOADING MODELS FROM CHECKPOINT")
        print("="*80)
        
        try:
            segment_models = joblib.load(models_checkpoint)
            print(f"Successfully loaded models for {len(segment_models)} segments")
        except Exception as e:
            print(f"ERROR loading models: {str(e)}")
            print("Falling back to training models")
            can_skip_model_training = False
    
    if not can_skip_model_training:
        print("\n" + "="*80)
        print("STEP 4: TRAINING SEGMENT-SPECIFIC MODELS WITH ENSEMBLE")
        print("="*80)
        
        # Count number of segments
        num_segments = len(np.unique(X_train_df['Segment']))
        print(f"Training models for {num_segments} segments...")
        
        # Prepare to store segment-specific models
        segment_models = {}
        
        # Train multiple models for each segment
        for segment_id in range(num_segments):
            print(f"\nTraining models for segment {segment_id}...")
            
            # Get segment data
            segment_mask = X_train_df['Segment'] == segment_id
            X_segment = X_train_df[segment_mask].drop(['Segment', 'Cluster'], axis=1, errors='ignore')
            y_segment = y_train_log_split[segment_mask]
            
            print(f"  Segment size: {len(X_segment)} samples")
            
            # Store ensemble of models for each segment
            segment_models[segment_id] = []
            
            # Choose model complexity based on segment
            if segment_id == num_segments - 1:  # Error segment
                # For error segment, train simpler models - Key change!
                model_types = ['normal', 'wide']  # Use simpler architecture
                epochs = 10  # Same epochs as regular segments
                print(f"  Using ensemble of simpler models for error-prone segment")
            else:
                model_types = ['normal', 'deep']
                epochs = 10
            
            # Train multiple models for ensemble
            for model_type in model_types:    
                # Build and train model
                model = fix_model_initialization(X_segment.shape[1], model_type)
                
                # Train with early stopping
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
                
                # Split into train/val within the segment
                segment_val_size = min(0.2, 100 / len(X_segment))  # Ensure enough validation samples
                
                print(f"  Training {model_type} model with {epochs} epochs...")
                
                history = model.fit(
                    X_segment, y_segment,
                    validation_split=segment_val_size,
                    epochs=epochs,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Add model to ensemble
                segment_models[segment_id].append(model)
                print(f"    {model_type} model trained with final loss: {history.history['loss'][-1]:.4f}")
        
        # Save models checkpoint
        joblib.dump(segment_models, models_checkpoint)
        print(f"Models checkpoint saved to {models_checkpoint}")
    
    # Step 5: Make Predictions - always run this step
    print("\n" + "="*80)
    print("STEP 5: MAKING ENSEMBLE PREDICTIONS WITH CALIBRATION")
    print("="*80)
    
    # Prepare test data
    if X_test_df is not None:
        # Store predictions for each segment separately
        segment_predictions = {}
        segment_counts = {}
        
        for segment_id, model_list in segment_models.items():
            # Get segment test data
            segment_mask = X_test_df['Segment'] == segment_id
            segment_count = segment_mask.sum()
            segment_counts[segment_id] = segment_count
            
            if segment_count > 0:
                X_segment_test = X_test_df[segment_mask].drop(['Segment', 'Cluster'], axis=1, errors='ignore')
                
                print(f"Making predictions for segment {segment_id} ({segment_count} samples)...")
                
                # Define a local function to avoid name collision
                def make_ensemble_prediction(models, X, segment_id=None):
                    return ensemble_predictions(models, X, segment_id)
                    
                # Use our local wrapper function instead
                segment_preds = make_ensemble_prediction(model_list, X_segment_test, segment_id=segment_id)
                
                # Apply calibration (especially for error segment)
                calibrated_preds = calibrate_predictions(segment_preds, segment_id=segment_id)
                
                # Store predictions for this segment
                segment_predictions[segment_id] = calibrated_preds
            else:
                print(f"No samples in segment {segment_id}")
        
        # Create full prediction array
        test_predictions = np.zeros(len(X_test_df))
        
        # Assign predictions from each segment
        for segment_id, preds in segment_predictions.items():
            segment_mask = X_test_df['Segment'] == segment_id
            test_predictions[segment_mask] = preds
            
        # Check for any zeros in the predictions (sanity check)
        zero_preds = (test_predictions == 0).sum()
        if zero_preds > 0:
            print(f"Warning: Found {zero_preds} zero predictions out of {len(test_predictions)}")
            # Fill zeros with median
            median_pred = np.median(test_predictions[test_predictions > 0])
            test_predictions[test_predictions == 0] = median_pred
            print(f"Replaced zero predictions with median value: {median_pred}")
        
        # Apply proper reverse scaling
        print("Reversing scaling transformations...")
        final_predictions = reverse_scaling(
            test_predictions, 
            checkpoint_dir=checkpoint_dir
        )
        
        # Analyze predictions before submission
        analyze_predictions(final_predictions, output_dir=results_dir)
        
        # Create submission file
        submission_path = os.path.join(results_dir, "improved_segment_submission.csv")
        create_submission(final_predictions, submission_path)
        print(f"Created submission with segment-based predictions: {submission_path}")

        # Also create a weighted ensemble submission (alternative approach)
        print("\nCreating weighted ensemble submission...")
        
        # Prepare segment prediction arrays for blending
        segment_arrays = []
        for segment_id in range(num_segments):
            segment_mask = X_test_df['Segment'] == segment_id
            segment_array = np.zeros(len(X_test_df))
            if segment_id in segment_predictions:
                segment_array[segment_mask] = segment_predictions[segment_id]
            segment_arrays.append(segment_array)
            
        # Create weighted blend - RENAME THIS VARIABLE to avoid name conflict
        blended_predictions = weighted_segment_blending(segment_arrays)
        
        # Convert from log space back to original scale
        ensemble_final = np.expm1(blended_predictions)
        
        # Create submission file for ensemble
        ensemble_path = os.path.join(results_dir, "weighted_ensemble_submission.csv")
        create_submission(ensemble_final, ensemble_path)
        print(f"Created submission with weighted ensemble approach: {ensemble_path}")
    
    print(f"\nPipeline completed in {time.time() - start_time:.2f} seconds")
    print("="*80)

if __name__ == "__main__":
    main()
