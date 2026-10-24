# Train the meta-selector model and save it to disk.
import os
import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_log_error

# Assuming BinManager is defined in a module named bin_manager
from bin_manager import BinManager
from bin_encoder import BinEncoder

def load_data(file_path):
    data = pd.read_csv(file_path)   
    return data

def save_bins(bins, file_path):
    """Convert numpy arrays to lists before saving"""
    serializable_bins = {}
    for feature, bin_info in bins.items():
        serializable_bins[feature] = {
            "edges": bin_info["edges"].tolist() if isinstance(bin_info["edges"], np.ndarray) else bin_info["edges"],
            "min_value": float(bin_info["edges"].min()) if isinstance(bin_info["edges"], np.ndarray) else float(min(bin_info["edges"])),
            "max_value": float(bin_info["edges"].max()) if isinstance(bin_info["edges"], np.ndarray) else float(max(bin_info["edges"]))
        }
    with open(file_path, 'w') as f:
        json.dump(serializable_bins, f, indent=2)

def load_bins(file_path):
    """Convert lists back to numpy arrays after loading"""
    with open(file_path, 'r') as f:
        bins_data = json.load(f)
    bins = {}
    for feature, bin_info in bins_data.items():
        bins[feature] = {
            "edges": np.array(bin_info["edges"], dtype=float),
            "min_value": bin_info["min_value"],
            "max_value": bin_info["max_value"]
        }
    return bins

def calculate_rmsle_score(actual, predicted):
    """Calculate RMSLE between actual and predicted values"""
    # Ensure no negative values
    actual = np.maximum(actual, 1e-6)
    predicted = np.maximum(predicted, 1e-6)
    return np.sqrt(mean_squared_log_error(actual, predicted))

def main():

    # Load submission information from /home/USER/Projects/ikaggle/AAA/submissions.JSON
    submissions = pd.read_json('/home/USER/Projects/ikaggle/AAA/submissions.JSON')

    # Load data from /home/USER/Projects/ikaggle/data/train.csv, /home/USER/Projects/ikaggle/data/validate.csv, /home/USER/Projects/ikaggle/data/test.csv
    train_data = load_data('/home/USER/Projects/ikaggle/data/train.csv')
    
    # Create bins using only training data features
    features_to_bin = [col for col in train_data.columns 
                      if col not in ['id', 'PremiumAmount']]
    
    # If bins file does not exist '/home/USER/Projects/ikaggle/AAA/submissions.JSON/bins.json'
    if not os.path.exists('/home/USER/Projects/ikaggle/AAA/bins.json'):
        print("No bin file found. Creating new bins.")
        bin_manager = BinManager()
        bins = bin_manager.create_bins(train_data, features_to_bin)
        save_bins(bins, '/home/USER/Projects/ikaggle/AAA/bins.json')
    else:
        print("Bins file found. Loading existing bins.") 
        bins = load_bins('/home/USER/Projects/ikaggle/AAA/bins.json')

    print("\nBin Summary:")
    for feature, bin_info in bins.items():
        print(f"\n{feature}:")
        print(f"  Number of edges: {len(bin_info['edges'])}")
        print(f"  Edges: {bin_info['edges']}")

    bin_manager = BinManager()
    # Create binary columns
    bin_encoder = BinEncoder(bin_manager)
    train_data_binary = bin_encoder.fit_transform(train_data, features_to_bin)
    
    print("\nBinary Column Summary:")
    print(f"Original features: {len(features_to_bin)}")
    print(f"Binary columns created: {len(bin_encoder.get_binary_columns())}")
    
    # Example of first few rows
    print("\nFirst few rows of binary features:")
    binary_cols = bin_encoder.get_binary_columns()
    print(train_data_binary[binary_cols].head())
    
    # Save binary encoded data
    train_data_binary.to_csv('/home/USER/Projects/ikaggle/data/train_binary.csv', index=False)

    # for each submission, load the train estimates to score by bins
    # for submission in submissions:
        # train_estimates = load_data(submission['train'])
        # Implement scoring logic to use bins and train_estimates
        # scores = score_bins(bins, train_estimates)
        # Save or process scores as needed

    # Load bitbins data
    bitbins_data = pd.read_csv('/home/USER/Projects/ikaggle/AAA/bitbins/train_bitbins.csv')
    
    # Load submissions info
    with open('/home/USER/Projects/ikaggle/AAA/submissions.JSON', 'r') as f:
        submissions = json.load(f)['submissions']
    
    # Step 1: Add submission estimates to bitbins data
    print("\nAdding submission estimates...")
    for submission in submissions:
        submission_id = submission['id']
        train_estimates = pd.read_csv(submission['train'])
        
        # Add the actual estimates as new columns
        bitbins_data[f'submission_{submission_id}'] = pd.merge(
            bitbins_data[['id']], 
            train_estimates[['id', 'Premium Amount']],
            on='id',
            how='left'
        )['Premium Amount']
        
        # Also calculate per-sample RMSLE scores
        merged = pd.merge(
            bitbins_data[['id', 'PremiumAmount']], 
            train_estimates[['id', 'Premium Amount']],
            on='id'
        )
        
        # Add per-sample scores
        bitbins_data[f'score_sub_{submission_id}'] = np.abs(
            np.log1p(merged['PremiumAmount']) - 
            np.log1p(merged['Premium Amount'])
        )
        
        # Print summary for this submission
        print(f"\nSubmission {submission_id}:")
        print(f"  Mean RMSLE: {merged[f'score_sub_{submission_id}'].mean():.6f}")
        print(f"  Estimates added as: submission_{submission_id}")
        print(f"  Scores added as: score_sub_{submission_id}")
    
    # Save enhanced bitbins data
    output_path = '/home/USER/Projects/ikaggle/AAA/bitbins/train_bitbins_with_submissions.csv'
    bitbins_data.to_csv(output_path, index=False)
    print(f"\nEnhanced bitbins saved to: {output_path}")
    
    # Print column summary
    estimate_cols = [col for col in bitbins_data.columns if col.startswith('submission_')]
    score_cols = [col for col in bitbins_data.columns if col.startswith('score_sub_')]
    
    print("\nData Summary:")
    print(f"Binary features: {len([col for col in bitbins_data.columns if '_bin_' in col])}")
    print(f"Submission estimates: {len(estimate_cols)}")
    print(f"Submission scores: {len(score_cols)}")
    
    # Show correlations between submissions
    print("\nSubmission Estimate Correlations:")
    corr = bitbins_data[estimate_cols].corr()
    print(corr)

    # Print summary statistics
    score_cols = [col for col in bitbins_data.columns if col.startswith('score_sub_')]
    print("\nScore Summary:")
    print(bitbins_data[score_cols].describe())

if __name__ == "__main__":
    main()