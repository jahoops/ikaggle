import pandas as pd
import numpy as np

def validate_submission(predictions, min_premium=100, max_premium=100000):
    """
    Validates submission predictions and ensures they are within reasonable bounds.
    
    Args:
        predictions: Array of predicted premium amounts
        min_premium: Minimum allowed premium amount 
        max_premium: Maximum allowed premium amount
        
    Returns:
        Cleaned predictions array with validated values
    """
    # Convert to float64 to avoid overflow
    predictions = np.array(predictions, dtype=np.float64)
    
    # Check for NaN/infinite values
    if np.any(~np.isfinite(predictions)):
        print(f"Warning: Found {np.sum(~np.isfinite(predictions))} invalid predictions")
        predictions = np.nan_to_num(predictions, nan=min_premium)

    # Apply bounds with proper casting
    predictions = np.clip(predictions, min_premium, max_premium)
    
    # Round to 2 decimal places
    predictions = np.round(predictions, 2)

    # Print detailed statistics
    print("\nSubmission Statistics:")
    percentiles = np.percentile(predictions, [0, 25, 50, 75, 100])
    print(f"Mean premium: ${predictions.mean():,.2f}")
    print(f"Median premium: ${percentiles[2]:,.2f}")
    print(f"Min premium: ${percentiles[0]:,.2f}")
    print(f"Max premium: ${percentiles[4]:,.2f}")
    print(f"\nPercentiles:")
    print(f"25th: ${percentiles[1]:,.2f}")
    print(f"75th: ${percentiles[3]:,.2f}")
    
    # Check distribution
    below_min = np.sum(predictions <= min_premium)
    above_max = np.sum(predictions >= max_premium)
    if below_min > 0 or above_max > 0:
        print(f"\nWarning:")
        print(f"Predictions at minimum ({min_premium}): {below_min:,} ({below_min/len(predictions):.2%})")
        print(f"Predictions at maximum ({max_premium}): {above_max:,} ({above_max/len(predictions):.2%})")

    return predictions
