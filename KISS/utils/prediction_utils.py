import numpy as np
import tensorflow as tf

def safe_predict(model, X, segment_id=None, batch_size=32, verbose=0):
    """
    Make predictions with additional safeguards against TF numerical issues
    """
    try:
        # First try normal prediction
        predictions = model.predict(X, batch_size=batch_size, verbose=verbose)
        
        # Check for NaNs or infinities
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print(f"Warning: Found NaN/Inf predictions in segment {segment_id}. Applying fallback.")
            # Switch to CPU for this prediction
            with tf.device('/CPU:0'):
                predictions = model.predict(X, batch_size=batch_size, verbose=verbose)
                
        # Final cleanup of any remaining NaNs or infinities
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=100.0, neginf=0.0)
        return predictions.flatten()
        
    except Exception as e:
        print(f"Error in prediction for segment {segment_id}: {str(e)}")
        # Return zeros as fallback
        return np.zeros(len(X))

def ensemble_predictions(models, X, segment_id=None, weights=None):
    """
    Create an ensemble of predictions from multiple models for more robustness
    """
    if not models:
        return np.zeros(len(X))
    
    all_preds = []
    for i, model in enumerate(models):
        try:
            preds = safe_predict(model, X, segment_id=f"{segment_id}-{i}")
            all_preds.append(preds)
        except Exception as e:
            print(f"Error in model {i} for segment {segment_id}: {str(e)}")
    
    if not all_preds:
        return np.zeros(len(X))
    
    # Use weights if provided, otherwise equal weighting
    if weights and len(weights) == len(all_preds):
        weighted_preds = np.zeros_like(all_preds[0])
        for i, preds in enumerate(all_preds):
            weighted_preds += weights[i] * preds
        return weighted_preds
    else:
        # Simple average
        return np.mean(all_preds, axis=0)

def calibrate_predictions(predictions, segment_id=None, calibration_factor=0.9):
    """
    Apply calibration to predictions based on segment
    
    This can help correct systematic biases in model predictions,
    which is especially important for error-prone segments.
    """
    # Default calibration - mild shrinkage toward the mean for stability
    adjusted_preds = predictions * calibration_factor
    
    # Special handling for error segment (assuming it's the last segment)
    if segment_id is not None and segment_id == 4:  # Error segment
        # More aggressive shrinkage for error segment
        adjusted_preds = predictions * (calibration_factor * 0.95)
        # Add a small constant to avoid extreme low values
        adjusted_preds += 0.05
        
    return adjusted_preds

def weighted_segment_blending(segment_predictions, segment_weights=None):
    """
    Create a weighted blend of segment predictions
    
    This can help when some segments are more reliable than others.
    """
    n_segments = len(segment_predictions)
    n_samples = len(segment_predictions[0])
    
    # Default to equal weights if none provided
    if segment_weights is None:
        segment_weights = [1.0 / n_segments] * n_segments
        # Downweight error segment (assuming it's the last one)
        if n_segments >= 4:  # If we have enough segments
            segment_weights[-1] *= 0.7  # Reduce weight of error segment
            # Normalize weights
            total = sum(segment_weights)
            segment_weights = [w / total for w in segment_weights]
    
    # Blend predictions
    blended = np.zeros(n_samples)
    for i in range(n_segments):
        blended += segment_weights[i] * segment_predictions[i]
        
    return blended
