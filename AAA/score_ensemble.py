import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from bin_encoder import BinEncoder
from bin_manager import BinManager
from ensemble_selector import safe_rmsle, EnsembleSelector  # Add EnsembleSelector import

def prepare_dataset(data: pd.DataFrame, bin_manager: BinManager, submissions: list, dataset_type: str = 'train') -> pd.DataFrame:
    """Process a dataset through the full pipeline"""
    print(f"\nPreparing {dataset_type} dataset...")
    print(f"Input shape: {data.shape}")
    
    # Create binary features
    bin_encoder = BinEncoder(bin_manager)
    features_to_bin = sorted([col for col in data.columns if col not in ['id', 'PremiumAmount']])
    
    if dataset_type == 'train':
        processed_data = bin_encoder.fit_transform(data, features_to_bin)
    else:
        # First fit on training data to get same binary columns
        train_data = pd.read_csv('/home/USER/Projects/ikaggle/data/train.csv')
        bin_encoder.fit_transform(train_data, features_to_bin)
        processed_data = bin_encoder.transform(data)
    
    # Add submission columns in consistent order
    for i in range(1, 10):
        col_name = f'submission_{i}'
        processed_data[col_name] = 0  # Initialize with zeros
        
        # Find matching submission if it exists
        sub = next((s for s in submissions if s['id'] == str(i)), None)
        if sub:
            sub_data = pd.read_csv(sub[dataset_type])
            processed_data[col_name] = pd.merge(
                processed_data[['id']], 
                sub_data[['id', 'Premium Amount']], 
                on='id', 
                how='left'
            )['Premium Amount'].fillna(0)
    
    print(f"Output shape: {processed_data.shape}")
    print("Features:", sorted(processed_data.columns))
    return processed_data

def main():
    # Load trained model
    model_path = Path('/home/USER/Projects/ikaggle/AAA/models/ensemble_selector.lgb')
    print(f"\nLoading model from: {model_path}")
    selector = joblib.load(model_path)
    print("Model feature count:", selector.model.n_features_)
    
    bin_manager = BinManager()
    with open('/home/USER/Projects/ikaggle/AAA/bins.json', 'r') as f:
        bins = json.load(f)
    bin_manager.feature_bins = {k: {"edges": np.array(v["edges"])} for k, v in bins.items()}
    
    # Load submissions info
    with open('/home/USER/Projects/ikaggle/AAA/submissions.JSON', 'r') as f:
        submissions = json.load(f)['submissions']
    
    # Load and process validation data
    print("\nProcessing validation data...")
    validate_data = pd.read_csv('/home/USER/Projects/ikaggle/data/validate.csv')
    validate_processed = prepare_dataset(validate_data, bin_manager, submissions, 'validate')
    
    # Make predictions
    predictions = selector.predict(validate_processed)
    
    # Calculate RMSLE
    actual = validate_data['PremiumAmount']
    rmsle = safe_rmsle(actual, predictions)
    print(f"\nValidation RMSLE: {rmsle:.6f}")
    
    # Compare with individual submissions
    print("\nIndividual Submission Scores:")
    for submission in submissions:
        sub_id = submission['id']
        sub_preds = validate_processed[f'submission_{sub_id}']
        sub_rmsle = safe_rmsle(actual, sub_preds)
        print(f"Submission {sub_id}: {sub_rmsle:.6f}")
    
    # Save validation predictions
    output_dir = Path('/home/USER/Projects/ikaggle/AAA/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame({
        'id': validate_data['id'],
        'PremiumAmount': predictions
    }).to_csv(output_dir / 'validation_predictions.csv', index=False)

if __name__ == "__main__":
    main()
