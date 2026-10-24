import pandas as pd
import numpy as np
from bin_manager import BinManager
from bin_encoder import BinEncoder
import json
import os
from pathlib import Path
from typing import Dict, Tuple

def save_metadata(metadata: dict, filepath: Path) -> None:
    """Save bin metadata for future reference"""
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)

def process_dataset(data: pd.DataFrame, bin_encoder: BinEncoder, 
                   features: list, is_train: bool = True) -> pd.DataFrame:
    """Process a dataset (train or test) into binary features"""
    # Create binary features
    binary_data = bin_encoder.transform(data, features)
    
    # Add id column
    binary_data['id'] = data['id']
    
    # Add target column for training data
    if is_train:
        binary_data['PremiumAmount'] = data['PremiumAmount']
    
    return binary_data

def main():
    # Setup paths
    base_dir = Path('/home/USER/Projects/ikaggle')
    data_dir = base_dir / 'data'
    output_dir = base_dir / 'AAA' / 'bitbins'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_data = pd.read_csv(data_dir / 'train.csv')
    test_data = pd.read_csv(data_dir / 'test.csv')
    
    # Identify features to bin
    features_to_bin = [col for col in train_data.columns 
                      if col not in ['id', 'PremiumAmount']]
    
    print(f"\nCreating bins for {len(features_to_bin)} features...")
    
    # Create bins using training data only
    bin_manager = BinManager(min_bin_size=1000)
    bins = bin_manager.create_bins(train_data, features_to_bin)
    
    # Save bin definitions
    print("\nSaving bin definitions...")
    bin_metadata = {
        'features': features_to_bin,
        'bins': {
            feature: {
                'edges': bin_info['edges'].tolist(),
                'min_value': float(bin_info['edges'].min()),
                'max_value': float(bin_info['edges'].max())
            }
            for feature, bin_info in bins.items()
        }
    }
    save_metadata(bin_metadata, output_dir / 'bin_metadata.json')
    
    # Create binary encoder
    print("\nConverting features to binary...")
    bin_encoder = BinEncoder(bin_manager)
    bin_encoder.fit(train_data, features_to_bin)
    
    # Process train and test data
    print("\nProcessing training data...")
    train_binary = process_dataset(train_data, bin_encoder, features_to_bin, is_train=True)
    
    print("Processing test data...")
    test_binary = process_dataset(test_data, bin_encoder, features_to_bin, is_train=False)
    
    # Save binary data
    train_binary.to_csv(output_dir / 'train_bitbins.csv', index=False)
    test_binary.to_csv(output_dir / 'test_bitbins.csv', index=False)
    
    # Save column information
    column_metadata = {
        'original_features': features_to_bin,
        'binary_columns': bin_encoder.get_binary_columns(),
        'id_column': 'id',
        'target_column': 'PremiumAmount',
        'train_shape': train_binary.shape,
        'test_shape': test_binary.shape,
        'n_binary_features': len(bin_encoder.get_binary_columns())
    }
    save_metadata(column_metadata, output_dir / 'column_metadata.json')
    
    # Print summary
    print("\nConversion Complete!")
    print(f"Original features: {len(features_to_bin)}")
    print(f"Binary features created: {len(bin_encoder.get_binary_columns())}")
    print("\nOutput shapes:")
    print(f"Train: {train_binary.shape}")
    print(f"Test: {test_binary.shape}")
    print(f"\nOutput files saved to: {output_dir}")
    print("Files created:")
    print(f"  - {output_dir/'bin_metadata.json'}")
    print(f"  - {output_dir/'column_metadata.json'}")
    print(f"  - {output_dir/'train_bitbins.csv'}")
    print(f"  - {output_dir/'test_bitbins.csv'}")
    
    # Show sample of binary columns
    print("\nExample binary column names:")
    print(bin_encoder.get_binary_columns()[:5])
    
    # Verify data consistency
    print("\nVerifying data consistency...")
    train_nulls = train_binary.isnull().sum().sum()
    test_nulls = test_binary.isnull().sum().sum()
    print(f"Null values - Train: {train_nulls}, Test: {test_nulls}")
    
    if train_nulls > 0 or test_nulls > 0:
        print("WARNING: Null values found in binary data!")

if __name__ == "__main__":
    main()
