import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

def reverse_scaling(predictions_log, processor=None, checkpoint_dir=None):
    """
    Properly reverse all scaling transforms that were applied to the target
    
    Args:
        predictions_log: Predictions in log space
        processor: InsuranceDataProcessor instance (optional)
        checkpoint_dir: Directory where scaling info might be saved (optional)
    
    Returns:
        Predictions in original scale
    """
    # First, we need to undo the log transformation
    print("Reversing log transformation...")
    predictions = np.expm1(predictions_log)
    
    # Create diagnostic histogram before any adjustments
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, alpha=0.7)
    plt.title("Predictions after log reverse transform")
    plt.xlabel("Predicted Premium")
    plt.ylabel("Count")
    plt.savefig("/home/USER/Projects/ikaggle/KISS/results/predictions_post_log_transform.png")
    plt.close()
    
    # If we have all predictions at 50.0, there's likely an issue with the model
    # or with the reverse scaling
    if np.sum(predictions < 50.0) > len(predictions) * 0.9:
        print("WARNING: Most predictions are below minimum premium threshold!")
        print(f"Min: {predictions.min():.4f}, Max: {predictions.max():.4f}, Mean: {predictions.mean():.4f}")
        
        # Try to load target scaling info if available
        scaling_info_path = os.path.join(checkpoint_dir, "target_scaling_info.pkl") if checkpoint_dir else None
        if scaling_info_path and os.path.exists(scaling_info_path):
            print(f"Found target scaling info at {scaling_info_path}")
            with open(scaling_info_path, 'rb') as f:
                scaling_info = pickle.load(f)
                
            # Apply scaling correction if needed
            if 'target_min' in scaling_info and 'target_max' in scaling_info:
                target_min = scaling_info['target_min']
                target_max = scaling_info['target_max']
                
                # Check if actual range is drastically different from expected
                actual_range = predictions.max() - predictions.min()
                expected_range = target_max - target_min
                
                if actual_range < expected_range * 0.1:  # If range is too small
                    print(f"Detected compressed prediction range: {actual_range} vs expected {expected_range}")
                    print("Applying range correction...")
                    
                    # Apply a histogram matching to expected distribution
                    # Create a simple stretching for now
                    current_min = predictions.min()
                    current_max = predictions.max()
                    
                    # Stretch to expected range
                    predictions = (predictions - current_min) / (current_max - current_min) * expected_range + target_min
    
    # Check for negative values (shouldn't happen with expm1 but just in case)
    negative_mask = predictions < 0
    if negative_mask.any():
        print(f"WARNING: Found {negative_mask.sum()} negative predictions. Setting to 0.0")
        predictions[negative_mask] = 0.0
    
    return predictions

def analyze_predictions(predictions, output_dir=None):
    """
    Analyze prediction distribution to identify issues
    """
    if output_dir is None:
        output_dir = "/home/USER/Projects/ikaggle/KISS/results"
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    stats = {
        'count': len(predictions),
        'min': float(predictions.min()),
        'max': float(predictions.max()),
        'mean': float(predictions.mean()),
        'median': float(np.median(predictions)),
        'std': float(predictions.std()),
        'zeros': int((predictions == 0).sum()),
        'min_premium_count': int((predictions == 50.0).sum()),
        'min_premium_pct': float((predictions == 50.0).sum() / len(predictions) * 100)
    }
    
    # Print statistics
    print("\nPrediction Analysis:")
    print(f"Count: {stats['count']}")
    print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
    print(f"Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, Std: {stats['std']:.4f}")
    print(f"Zero predictions: {stats['zeros']}")
    print(f"Minimum premium predictions: {stats['min_premium_count']} ({stats['min_premium_pct']:.2f}%)")
    
    # Create distribution plot
    plt.figure(figsize=(12, 6))
    plt.hist(predictions, bins=50, alpha=0.7)
    plt.axvline(x=50.0, color='r', linestyle='--', label='Min Premium')
    plt.title("Prediction Distribution")
    plt.xlabel("Predicted Premium")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
    plt.close()
    
    # Log transform for better visualization of tail
    plt.figure(figsize=(12, 6))
    plt.hist(np.log1p(predictions), bins=50, alpha=0.7)
    plt.axvline(x=np.log1p(50.0), color='r', linestyle='--', label='Min Premium')
    plt.title("Log-Transformed Prediction Distribution")
    plt.xlabel("Log(Premium + 1)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, "prediction_log_distribution.png"))
    plt.close()
    
    return stats

def extract_scaling_info_from_processor(processor):
    """
    Extract scaling information from the data processor
    """
    scaling_info = {
        'target_min': float(processor.target.min()),
        'target_max': float(processor.target.max()),
        'target_mean': float(processor.target.mean()),
        'target_median': float(processor.target.median()),
        'target_std': float(processor.target.std()),
        'has_scalers': hasattr(processor, 'scalers') and len(processor.scalers) > 0
    }
    
    return scaling_info
