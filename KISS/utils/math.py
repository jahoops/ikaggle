import numpy as np
import warnings

def safe_rmsle(y_true, y_pred):
    """
    Calculate RMSLE (Root Mean Squared Logarithmic Error) safely.
    Handles negative, zero, NaN, and infinite values properly.
    """
    # Handle special cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return float('inf')
        
    # Make copies to avoid modifying originals
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Ensure equal lengths
    if len(y_true) != len(y_pred):
        warnings.warn(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    # Handle invalid values
    # Replace negative values with small positive
    y_true = np.maximum(y_true, 1e-6)
    y_pred = np.maximum(y_pred, 1e-6)
    
    # Handle NaNs by replacing with mean
    true_mean = np.nanmean(y_true)
    pred_mean = np.nanmean(y_pred)
    
    y_true = np.where(np.isnan(y_true), true_mean, y_true)
    y_pred = np.where(np.isnan(y_pred), pred_mean, y_pred)
    
    # Handle infinities by replacing with large values
    y_true = np.where(np.isinf(y_true), 1e9, y_true)
    y_pred = np.where(np.isinf(y_pred), 1e9, y_pred)
    
    # Clip values to reasonable range
    y_true = np.clip(y_true, 1e-6, 1e9)
    y_pred = np.clip(y_pred, 1e-6, 1e9)
    
    # Calculate RMSLE
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    mean_squared_log_error = np.mean(log_diff ** 2)
    rmsle = np.sqrt(mean_squared_log_error)
    
    return rmsle