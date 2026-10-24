"""
Enhanced version of the pipeline that adds a segment for poorly predicted samples.
With checkpointing support to skip steps that have already been completed.
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
from utils.prediction_utils import safe_predict  # Add this import

def create_submission(predictions, output_path, min_premium=50.0):
    # Ensure predictions are positive
    predictions = np.maximum(predictions, 0.0)
    
    # Apply minimum premium threshold
    predictions = np.maximum(predictions, min_premium)
    
    # Create submission data
    submission_df = pd.DataFrame({
        'id': range(1200000, 1200000 + len(predictions)),
        'Premium Amount': predictions
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    return submission_df

def main():
    # Setup argument parser to control which steps to run
    parser = argparse.ArgumentParser(description='Run enhanced pipeline with error segment')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-base-segmentation', action='store_true', help='Skip base segmentation')
    parser.add_argument('--skip-error-analysis', action='store_true', help='Skip error analysis')
    parser.add_argument('--skip-model-training', action='store_true', help='Skip model training')
    parser.add_argument('--only-predictions', action='store_true', help='Only make predictions using existing models')
    args = parser.parse_args()
    
    # Start timing the pipeline
    start_time = time.time()
    
    print("Starting insurance premium prediction pipeline with ERROR ENHANCEMENT...")
    
    # Setup directories
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
    results_dir = os.path.join(os.path.dirname(__file__), "results", "enhanced_pipeline")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Define checkpoint paths
    data_checkpoint = os.path.join(checkpoint_dir, "processed_data.pkl")
    base_segment_checkpoint = os.path.join(checkpoint_dir, "base_segments.pkl")
    enhanced_segment_checkpoint = os.path.join(checkpoint_dir, "enhanced_segments.pkl")
    models_checkpoint = os.path.join(checkpoint_dir, "segment_models.pkl")
    
    # Initialize validator
    validator = PipelineValidator()
    
    # Step 1: Data Loading and Processing
    if args.only_predictions:
        print("\nLoading processed data from checkpoint...")
        with open(data_checkpoint, 'rb') as f:
            data = pickle.load(f)
            X_train_df = data['X_train']
            X_val_df = data['X_val'] 
            X_test_df = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_train_log_split = data['y_train_log']
            y_val_log = data['y_val_log']
    elif args.skip_preprocessing and os.path.exists(data_checkpoint):
        print("\nSkipping data preprocessing, loading from checkpoint...")
        with open(data_checkpoint, 'rb') as f:
            data = pickle.load(f)
            X_train_df = data['X_train']
            X_val_df = data['X_val'] 
            X_test_df = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_train_log_split = data['y_train_log']
            y_val_log = data['y_val_log']
    else:
        print("\n" + "="*80)
        print("STEP 1: LOADING AND PROCESSING DATA")
        print("="*80)
        
        # Define file paths
        train_path = os.path.join(DATA_DIR, "train_raw.csv")
        test_path = os.path.join(DATA_DIR, "contest_raw.csv")
        
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

    # Step 2: Base Segmentation
    if args.only_predictions:
        print("\nLoading enhanced segments from checkpoint...")
        with open(enhanced_segment_checkpoint, 'rb') as f:
            segments_data = pickle.load(f)
            enhanced_segmenter = segments_data['segmenter']
            X_train_df['Segment'] = segments_data['train_segments']
            X_val_df['Segment'] = segments_data['val_segments'] 
            X_test_df['Segment'] = segments_data['test_segments']
    elif args.skip_base_segmentation and args.skip_error_analysis and os.path.exists(enhanced_segment_checkpoint):
        print("\nSkipping segmentation, loading enhanced segments from checkpoint...")
        with open(enhanced_segment_checkpoint, 'rb') as f:
            segments_data = pickle.load(f)
            enhanced_segmenter = segments_data['segmenter']
            X_train_df['Segment'] = segments_data['train_segments']
            X_val_df['Segment'] = segments_data['val_segments'] 
            X_test_df['Segment'] = segments_data['test_segments']
    else:
        # Run base segmentation if needed
        base_segments = None
        if args.skip_base_segmentation and os.path.exists(base_segment_checkpoint):
            print("\nSkipping base segmentation, loading from checkpoint...")
            with open(base_segment_checkpoint, 'rb') as f:
                segment_data = pickle.load(f)
                segmenter = segment_data['segmenter']
                base_segments = segment_data['segments']
        else:
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
        if args.skip_error_analysis and os.path.exists(enhanced_segment_checkpoint):
            print("\nSkipping error analysis, loading enhanced segments from checkpoint...")
            with open(enhanced_segment_checkpoint, 'rb') as f:
                segments_data = pickle.load(f)
                enhanced_segmenter = segments_data['segmenter']
                X_train_df['Segment'] = segments_data['train_segments']
                X_val_df['Segment'] = segments_data['val_segments']
                X_test_df['Segment'] = segments_data['test_segments']
        else:
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
            
            # Enhanced segmentation with error analysis
            print("Creating enhanced segments including error-prone group...")
            error_analyzer = ErrorAnalyzer(error_threshold=0.85)  # Top 15% of errors
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
    if args.only_predictions or args.skip_model_training:
        if os.path.exists(models_checkpoint):
            print("\nSkipping model training, loading models from checkpoint...")
            segment_models = joblib.load(models_checkpoint)
        else:
            raise ValueError("Cannot skip model training: models checkpoint not found")
    else:
        print("\n" + "="*80)
        print("STEP 4: TRAINING SEGMENT-SPECIFIC MODELS (INCLUDING ERROR SEGMENT)")
        print("="*80)
        
        # Now the num_segments is 5 (original 4 + error segment)
        num_segments = len(np.unique(X_train_df['Segment']))
        print(f"Training models for {num_segments} segments...")
        
        # Prepare to store segment-specific models
        segment_models = {}
        
        # Train a model for each segment
        for segment_id in range(num_segments):
            print(f"\nTraining model for segment {segment_id}...")
            
            # Get segment data
            segment_mask = X_train_df['Segment'] == segment_id
            X_segment = X_train_df[segment_mask].drop(['Segment', 'Cluster'], axis=1, errors='ignore')
            y_segment = y_train_log_split[segment_mask]
            
            # Choose model complexity based on segment
            if segment_id == num_segments - 1:  # Error segment
                model_type = 'error_segment'
                epochs = 15  # More epochs for error segment
                print(f"Using specialized configuration for error-prone segment")
            else:
                model_type = 'normal'
                epochs = 10
                
            # Build and train model
            model = build_insurance_model(X_segment.shape[1], model_type)
            
            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Split into train/val within the segment
            segment_val_size = min(0.2, 100 / len(X_segment))  # Ensure enough validation samples
            
            history = model.fit(
                X_segment, y_segment,
                validation_split=segment_val_size,
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Save the model
            segment_models[segment_id] = model
            print(f"Segment {segment_id} model trained with final loss: {history.history['loss'][-1]:.4f}")
        
        # Save models checkpoint
        joblib.dump(segment_models, models_checkpoint)
        print(f"Models checkpoint saved to {models_checkpoint}")
    
    # Step 5: Make Predictions - always run this step
    print("\n" + "="*80)
    print("STEP 5: MAKING PREDICTIONS WITH SEGMENT-SPECIFIC MODELS")
    print("="*80)
    
    # Prepare test data
    if X_test_df is not None:
        # Make predictions for each segment
        test_predictions = np.zeros(len(X_test_df))
        
        for segment_id, model in segment_models.items():
            # Get segment test data
            segment_mask = X_test_df['Segment'] == segment_id
            if segment_mask.sum() > 0:
                X_segment_test = X_test_df[segment_mask].drop(['Segment', 'Cluster'], axis=1, errors='ignore')
                
                # Use safe_predict instead of direct prediction
                segment_preds = safe_predict(model, X_segment_test, segment_id=segment_id, verbose=0)
                
                # Add to overall predictions
                test_predictions[segment_mask] = segment_preds
                
                print(f"Made {segment_mask.sum()} predictions for segment {segment_id}")
        
        # Convert from log space back to original scale
        final_predictions = np.expm1(test_predictions)
        
        # Create submission file
        submission_path = os.path.join(results_dir, "enhanced_segment_submission.csv")
        create_submission(final_predictions, submission_path)
    
    print(f"Pipeline completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
