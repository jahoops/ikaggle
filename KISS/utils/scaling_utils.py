"""
Utility functions for scaling and unscaling data
"""

import numpy as np
import pickle
import os

def load_target_scaler(checkpoint_dir):
    """
    Load target scaler from checkpoint directory
    """
    scaler_path = os.path.join(checkpoint_dir, "target_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

def unscale_predictions(predictions, checkpoint_dir):
    """
    Unscale predictions if a target scaler exists
    """
    # Load target scaler if it exists
    target_scaler = load_target_scaler(checkpoint_dir)
    
    if target_scaler is not None:
        print("Unscaling predictions using saved target scaler...")
        # Reshape for scikit-learn's inverse_transform
        predictions_reshaped = predictions.reshape(-1, 1)
        # Apply inverse transform
        unscaled_preds = target_scaler.inverse_transform(predictions_reshaped).flatten()
        return unscaled_preds
    
    # If no scaler found, return original predictions
    return predictions
