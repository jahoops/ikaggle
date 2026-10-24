import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def fix_premiums(input_file, output_file=None, target_mean=500.0):
    """
    Fix the premium predictions to be realistic insurance values
    
    Args:
        input_file: Path to the input submission CSV
        output_file: Path to save the fixed submission (default: adds '_fixed' to original)
        target_mean: Target mean premium amount (default: $500)
    """
    print(f"Loading submission from {input_file}")
    df = pd.read_csv(input_file)
    
    # Extract predictions
    pred_col = 'Premium Amount'
    preds = df[pred_col].values
    
    # Analyze original predictions
    print("\nOriginal Prediction Statistics:")
    print(f"Count: {len(preds)}")
    print(f"Min: ${np.min(preds):.2f}")
    print(f"Max: ${np.max(preds):.2f}")
    print(f"Mean: ${np.mean(preds):.2f}")
    print(f"Median: ${np.median(preds):.2f}")
    print(f"Zero values: {np.sum(preds == 0)} ({100*np.sum(preds == 0)/len(preds):.1f}%)")
    print(f"Values < 1: {np.sum((preds > 0) & (preds < 1))} ({100*np.sum((preds > 0) & (preds < 1))/len(preds):.1f}%)")
    
    # Calculate appropriate scaling factor
    non_zero_mean = np.mean(preds[preds > 0])
    scaling_factor = target_mean / max(non_zero_mean, 0.01)  # Avoid division by zero
    print(f"\nDetected scaling factor: {scaling_factor:.2f}")
    
    # Create fixed predictions
    fixed_preds = preds * scaling_factor
    
    # Apply minimum premium for true insurance premiums (even zeros)
    # Small non-zero predictions might represent very low-risk policies
    min_premium = 75.0  # Min reasonable insurance premium
    
    # Apply minimum premium to very small values
    small_mask = fixed_preds < min_premium
    small_count = np.sum(small_mask)
    
    if small_count > 0:
        # Use rank-based approach for small values to maintain relative ordering
        small_values = fixed_preds[small_mask]
        ranks = small_values.argsort().argsort() / max(1, len(small_values)-1)  # 0 to 1
        min_adjusted = min_premium/2 + min_premium/2 * ranks  # Range from min_premium/2 to min_premium
        
        # Replace small values with rank-adjusted minimums
        fixed_preds[small_mask] = min_adjusted
        print(f"Applied minimum premium adjustment to {small_count} values ({100*small_count/len(preds):.1f}%)")
    
    # Analyze fixed predictions
    print("\nFixed Prediction Statistics:")
    print(f"Min: ${np.min(fixed_preds):.2f}")
    print(f"Max: ${np.max(fixed_preds):.2f}")
    print(f"Mean: ${np.mean(fixed_preds):.2f}")
    print(f"Median: ${np.median(fixed_preds):.2f}")
    
    # Create output path if not provided
    if output_file is None:
        name_parts = os.path.splitext(input_file)
        output_file = f"{name_parts[0]}_fixed{name_parts[1]}"
    
    # Save fixed predictions
    fixed_df = df.copy()
    fixed_df[pred_col] = fixed_preds
    fixed_df.to_csv(output_file, index=False)
    print(f"\nSaved fixed predictions to {output_file}")
    
    # Create visualization of before/after
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    sns.histplot(preds, bins=50, kde=True)
    plt.title("Original Predictions")
    plt.xlabel("Premium Amount ($)")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    sns.histplot(fixed_preds, bins=50, kde=True)
    plt.title("Fixed Predictions")
    plt.xlabel("Premium Amount ($)")
    
    plt.tight_layout()
    viz_file = f"{os.path.splitext(output_file)[0]}_comparison.png"
    plt.savefig(viz_file)
    print(f"Saved visualization to {viz_file}")
    
    return output_file

if __name__ == "__main__":
    # Fix the ensemble submission
    input_file = '/home/USER/Projects/ikaggle/KISS/results/final_group_ensemble_submission.csv'
    output_file = '/home/USER/Projects/ikaggle/KISS/results/final_submission.csv'
    
    fix_premiums(input_file, output_file)
