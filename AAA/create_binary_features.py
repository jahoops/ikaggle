import pandas as pd
from bin_encoder import BinEncoder
from bin_manager import BinManager
import os

def main():
    # Load the original training data
    print("Loading training data...")
    train_data = pd.read_csv('/home/USER/Projects/ikaggle/data/train.csv')
    
    # Load the bin definitions if they exist
    if not os.path.exists('/home/USER/Projects/ikaggle/AAA/bins.json'):
        print("Error: Bins must be created first using score_by_bin.py")
        return
        
    # Get features to bin (all except id and target)
    features_to_bin = [col for col in train_data.columns 
                      if col not in ['id', 'PremiumAmount']]
    
    print(f"\nConverting {len(features_to_bin)} features to binary...")
    
    # Create binary features
    bin_encoder = BinEncoder()
    binary_data = bin_encoder.fit_transform(train_data, features_to_bin)
    
    # Keep original id and target
    binary_data['id'] = train_data['id']
    binary_data['PremiumAmount'] = train_data['PremiumAmount']
    
    # Save binary version
    output_path = '/home/USER/Projects/ikaggle/data/train_binary.csv'
    binary_data.to_csv(output_path, index=False)
    
    # Print summary
    print("\nBinary Feature Creation Summary:")
    print(f"Original features: {len(features_to_bin)}")
    print(f"Binary features created: {len(bin_encoder.get_binary_columns())}")
    print(f"Total columns in output: {binary_data.shape[1]}")
    print(f"Saved to: {output_path}")
    
    # Show sample of binary columns
    print("\nFirst 5 binary column names:")
    print(bin_encoder.get_binary_columns()[:5])
    
if __name__ == "__main__":
    main()
