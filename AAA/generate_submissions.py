import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from bin_encoder import BinEncoder
from bin_manager import BinManager
from ensemble_selector import safe_rmsle, EnsembleSelector  # Import EnsembleSelector class

def load_data_and_model():
    """Load the trained model and required data"""
    model_path = Path('/home/USER/Projects/ikaggle/AAA/models/ensemble_selector.lgb')
    print(f"Loading model from: {model_path}")
    selector = joblib.load(model_path)
    
    # Load bins data
    bin_manager = BinManager()
    with open('/home/USER/Projects/ikaggle/AAA/bins.json', 'r') as f:
        bins = json.load(f)
    bin_manager.feature_bins = {k: {"edges": np.array(v["edges"])} for k, v in bins.items()}
    
    # Load submissions info
    with open('/home/USER/Projects/ikaggle/AAA/submissions.JSON', 'r') as f:
        submissions = json.load(f)['submissions']
        
    return selector, submissions, bin_manager

def process_dataset(data: pd.DataFrame, bin_manager: BinManager, submissions: list, dataset_type: str) -> pd.DataFrame:
    """Process dataset through binary encoding and add submission predictions"""
    # First create binary features
    bin_encoder = BinEncoder(bin_manager)
    features_to_bin = [col for col in data.columns if col not in ['id', 'PremiumAmount']]
    
    # Always fit on training data first
    print("Loading training data for feature encoding...")
    train_data = pd.read_csv('/home/USER/Projects/ikaggle/data/train.csv')
    bin_encoder.fit_transform(train_data, features_to_bin)
    
    # Then transform the target dataset
    print(f"Transforming {dataset_type} data...")
    processed = bin_encoder.transform(data)
    print(f"Created {len(bin_encoder.get_binary_columns())} binary features")
    
    # Add submission predictions
    print("Adding submission predictions...")
    for submission in submissions:
        sub_id = submission['id']
        sub_data = pd.read_csv(submission[dataset_type])
        processed[f'submission_{sub_id}'] = pd.merge(
            processed[['id']], 
            sub_data[['id', 'Premium Amount']], 
            on='id', 
            how='left'
        )['Premium Amount'].fillna(0)
    
    print(f"Final processed shape: {processed.shape}")
    return processed

def evaluate_dataset(selector, data: pd.DataFrame, dataset_type: str, bin_manager: BinManager, submissions: list) -> float:
    """Evaluate model performance on a dataset"""
    print(f"\n{'='*50}")
    print(f"Processing {dataset_type} dataset")
    print(f"{'='*50}")
    
    print(f"\nInput shape: {data.shape}")
    processed = process_dataset(data, bin_manager, submissions, dataset_type)
    print(f"Processed shape: {processed.shape}")
    
    print("\nGenerating predictions...")
    predictions = selector.predict(processed)
    print(f"Generated {len(predictions)} predictions")
    
    if 'PremiumAmount' in data.columns:
        rmsle = safe_rmsle(data['PremiumAmount'], predictions)
        print(f"\n{dataset_type.capitalize()} RMSLE: {rmsle:.6f}")
        
        print(f"\n{dataset_type.capitalize()} Individual Submission Scores:")
        submission_scores = []
        for submission in submissions:
            sub_id = submission['id']
            sub_preds = processed[f'submission_{sub_id}']
            sub_rmsle = safe_rmsle(data['PremiumAmount'], sub_preds)
            submission_scores.append((sub_id, sub_rmsle))
            print(f"Submission {sub_id}: {sub_rmsle:.6f}")
        
        # Print best submission
        best_sub_id, best_score = min(submission_scores, key=lambda x: x[1])
        print(f"\nBest submission: {best_sub_id} (RMSLE: {best_score:.6f})")
        print(f"Ensemble improvement: {best_score - rmsle:.6f}")
        
        return rmsle
    return None

def main():
    print("\nLoading model and data...")
    selector, submissions, bin_manager = load_data_and_model()
    
    # Process and evaluate training data
    train_data = pd.read_csv('/home/USER/Projects/ikaggle/data/train.csv')
    train_rmsle = evaluate_dataset(selector, train_data, 'train', bin_manager, submissions)
    
    # Process and evaluate validation data
    validate_data = pd.read_csv('/home/USER/Projects/ikaggle/data/validate.csv')
    validate_rmsle = evaluate_dataset(selector, validate_data, 'validate', bin_manager, submissions)
    
    print("\nPerformance Summary")
    print("="*50)
    print(f"Training RMSLE:   {train_rmsle:.6f}")
    print(f"Validation RMSLE: {validate_rmsle:.6f}")
    print(f"Difference:       {abs(validate_rmsle - train_rmsle):.6f}")
    
    # Process test data
    print("\nProcessing test data...")
    test_data = pd.read_csv('/home/USER/Projects/ikaggle/data/test.csv')
    test_processed = process_dataset(test_data, bin_manager, submissions, 'test')
    test_pred = selector.predict(test_processed)
    
    # Save predictions
    output_dir = Path('/home/USER/Projects/ikaggle/AAA/submissions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({
        'id': test_data['id'],
        'Premium Amount': test_pred
    }).to_csv(output_dir / 'ensemble_submission.csv', index=False)
    
    print(f"\nSubmission saved to: {output_dir/'ensemble_submission.csv'}")

if __name__ == "__main__":
    main()
