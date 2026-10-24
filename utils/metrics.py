import numpy as np
from sklearn.metrics import mean_squared_log_error

def safe_rmsle(y_true, y_pred):
    # Drop NaNs or inf values
    valid_mask = ~np.isnan(y_pred) & ~np.isinf(y_pred)
    return np.sqrt(mean_squared_log_error(y_true[valid_mask], y_pred[valid_mask]))