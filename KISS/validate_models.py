"""
Script to validate models and diagnose prediction issues
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
from utils.model_fixes import diagnose_predictions

def main():
    print("Model and Prediction Validation Tool")
    print("="*50)
    
    # Define paths
    results_dir = os.path.join(os.path.dirname(__file__), "results", "improved_pipeline")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    data_checkpoint = os.path.join(checkpoint_dir, "processed_data.pkl")
    models_checkpoint = os.path.join(checkpoint_dir, "segment_models.pkl")
    
    # Load data
    print("\nLoading data...")
    with open(data_checkpoint, 'rb') as f:
        data = pickle.load(f)
        X_train_df = data['X_train']
        X_val_df = data['X_val'] 
        X_test_df = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_train_log = data['y_train_log']
        y_val_log = data['y_val_log']
    
    # Load models
    print("\nLoading models...")
    segment_models = joblib.load(models_checkpoint)
    
    # Test on validation data
    print("\nTesting model predictions on validation data...")
    
    # Create dataframe to track each model's performance
    model_stats = pd.DataFrame(columns=['Segment', 'Model', 'Mean', 'Median', 'Min', 'Max', 'Zeros', 'NaNs'])
    
    for segment_id, model_list in segment_models.items():
        # Get segment data from validation set
        segment_mask = X_val_df['Segment'] == segment_id
        X_segment_val = X_val_df[segment_mask].drop(['Segment', 'Cluster'], axis=1, errors='ignore')
        y_segment_val = y_val_log[segment_mask]
        
        if len(X_segment_val) > 0:
            print(f"\nSegment {segment_id}: {len(X_segment_val)} samples")
            
            # Test each model in the ensemble
            for i, model in enumerate(model_list):
                try:
                    # Make predictions
                    val_preds_log = model.predict(X_segment_val, verbose=0).flatten()
                    
                    # Convert from log space
                    val_preds = np.expm1(val_preds_log)
                    
                    # Diagnose predictions
                    stats = diagnose_predictions(
                        val_preds, 
                        original=np.expm1(y_segment_val), 
                        title=f"Segment {segment_id} Model {i}"
                    )
                    
                    # Add to stats dataframe
                    model_stats = model_stats._append({
                        'Segment': segment_id,
                        'Model': i,
                        'Mean': stats['mean'],
                        'Median': stats['median'],
                        'Min': stats['min'],
                        'Max': stats['max'],
                        'Zeros': stats['zeros'],
                        'NaNs': stats['nans']
                    }, ignore_index=True)
                    
                except Exception as e:
                    print(f"Error testing model {i} for segment {segment_id}: {str(e)}")
    
    # Save model stats
    model_stats.to_csv(os.path.join(results_dir, "model_validation_stats.csv"), index=False)
    print(f"\nModel validation stats saved to {os.path.join(results_dir, 'model_validation_stats.csv')}")
    
    # Print table of segment statistics
    print("\nModel Performance Summary:")
    print(model_stats.groupby('Segment').mean()[['Mean', 'Median', 'Min', 'Max']])
    
    print("\nValidation complete!")

if __name__ == "__main__":
    main()
