import numpy as np
import pandas as pd

def load_test_data():
    """
    Minimal test data loader.
    Returns:
        X_full: pandas DataFrame with dummy feature data.
        y_full: numpy array with dummy target values.
    """
    X_full = pd.DataFrame({
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': [5, 4, 3, 2, 1]
    })
    y_full = np.array([10, 20, 30, 40, 50])
    return X_full, y_full
