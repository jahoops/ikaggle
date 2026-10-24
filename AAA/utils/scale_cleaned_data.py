import pandas as pd
import numpy as np
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS, get_gene_length

def scale_data():
    # Define features to scale - assuming these are in FEATURE_CONFIGS
    features_to_scale = [feat for feat in FEATURE_CONFIGS if FEATURE_CONFIGS[feat].get('scale', True)]

    # Load and process each file
    for file in ['train', 'validate', 'test', 'testing']:
        print(f"Processing {file} dataset...")
        try:
            # Read the data
            myfile = pd.read_csv(f'/home/USER/Projects/ikaggle/data/processed/cleaned_{file}.csv')
            
            # Create copy of original data
            scaled_data = myfile.copy()
            
            # Only scale selected features
            for feature in features_to_scale:
                if feature in myfile.columns:
                    min_val = myfile[feature].min()
                    max_val = myfile[feature].max()
                    if max_val - min_val != 0:  # Prevent division by zero
                        scaled_data[feature] = (myfile[feature] - min_val) / (max_val - min_val)
                    else:
                        print(f"Warning: {feature} has no variance, skipping scaling")
            
            # Save scaled data
            scaled_data.to_csv(f'/home/USER/Projects/ikaggle/AAA/data/scaled_{file}.csv', index=False)
            print(f"Successfully scaled and saved {file} dataset")
            
        except Exception as e:
            print(f"Error processing {file} dataset: {str(e)}")

if __name__ == "__main__":
    scale_data()
