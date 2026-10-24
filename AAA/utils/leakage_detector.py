import pandas as pd
import numpy as np
from scipy import stats

def detect_leakage_sources(X, y, threshold=0.9):
    """Detect potential sources of data leakage"""
    leakage_report = []
    
    # 1. Direct value matching
    for col in X.columns:
        # Check if column contains exact target values
        exact_match = np.sum(np.abs(X[col] - y) < 1e-6) / len(y)
        if exact_match > 0.01:  # More than 1% exact matches
            leakage_report.append(f"WARNING: {col} contains exact target values ({exact_match:.2%} matches)")
            
        # Check scaled/transformed matches
        y_norm = (y - y.mean()) / y.std()
        x_norm = (X[col] - X[col].mean()) / X[col].std()
        scale_match = np.sum(np.abs(x_norm - y_norm) < 0.01) / len(y)
        if scale_match > 0.1:  # More than 10% scaled matches
            leakage_report.append(f"WARNING: {col} matches target after scaling ({scale_match:.2%} matches)")
    
    # 2. Check for high correlations with target
    for col in X.columns:
        corr = stats.spearmanr(X[col], y)[0]
        if abs(corr) > threshold:
            leakage_report.append(f"WARNING: {col} has suspicious correlation with target ({corr:.3f})")
    
    # 3. Check for columns that might be target transformations
    log_y = np.log1p(y)
    sqrt_y = np.sqrt(y)
    for col in X.columns:
        # Check log transform
        log_corr = stats.spearmanr(X[col], log_y)[0]
        if abs(log_corr) > threshold:
            leakage_report.append(f"WARNING: {col} correlates with log(target) ({log_corr:.3f})")
        
        # Check sqrt transform
        sqrt_corr = stats.spearmanr(X[col], sqrt_y)[0]
        if abs(sqrt_corr) > threshold:
            leakage_report.append(f"WARNING: {col} correlates with sqrt(target) ({sqrt_corr:.3f})")
    
    return leakage_report

