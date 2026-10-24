import numpy as np
import pandas as pd
from scipy import stats

def inspect_feature_correlations(X, y, threshold=0.3):
    """Check for suspiciously high correlations with target"""
    correlations = {}
    suspicious_features = []
    
    for col in X.columns:
        corr = stats.spearmanr(X[col], y)[0]
        correlations[col] = corr
        if abs(corr) > threshold:
            suspicious_features.append((col, corr))
    
    print("\nHigh correlation features:")
    for feature, corr in sorted(suspicious_features, key=lambda x: abs(x[1]), reverse=True):
        print(f"{feature}: {corr:.3f}")
        
    return correlations, suspicious_features

def check_data_leakage(train_df, test_df):
    """Check for potential data leakage between train and test"""
    print("\nChecking for data leakage...")
    
    # Check for duplicate columns
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common_cols = train_cols.intersection(test_cols)
    
    print(f"\nCommon columns: {len(common_cols)}")
    
    # Check for identical values
    for col in common_cols:
        unique_train = set(train_df[col].unique())
        unique_test = set(test_df[col].unique())
        overlap = len(unique_train.intersection(unique_test))
        if overlap > 0:
            overlap_pct = (overlap / len(unique_train)) * 100
            if overlap_pct > 50:
                print(f"\nWarning: High value overlap in {col}")
                print(f"Overlap: {overlap_pct:.1f}%")
                print("Sample overlapping values:", 
                      list(unique_train.intersection(unique_test))[:5])

def inspect_derived_features(X, y, suspicious_threshold=0.95):
    """Check for features that might be derived from target"""
    print("\nChecking for derived features...")
    
    for col in X.columns:
        # Check for perfect/near-perfect functional relationships
        if X[col].nunique() > 1:  # Skip constant columns
            # Try common transformations
            transforms = {
                'log': np.log1p(y),
                'sqrt': np.sqrt(y),
                'square': y**2,
                'inverse': 1/y,
                'scaled': y/1000
            }
            
            for name, transformed in transforms.items():
                corr = abs(stats.spearmanr(X[col], transformed)[0])
                if corr > suspicious_threshold:
                    print(f"\nWarning: {col} highly correlated with {name}(target)")
                    print(f"Correlation: {corr:.3f}")
                    print("Sample values:")
                    print(pd.DataFrame({
                        'feature': X[col].head(),
                        'target': y.head(),
                        'transformed': transformed.head()
                    }))

def inspect_numerical_relationships(X, y, threshold=0.95):
    """Check for suspicious numerical relationships with target"""
    print("\nChecking numerical relationships...")
    
    suspicious_features = []
    for col in X.columns:
        # Check exact matches after scaling
        y_scaled = y / y.mean()
        x_scaled = X[col] / X[col].mean()
        
        diff = np.abs(y_scaled - x_scaled)
        match_ratio = np.mean(diff < 0.01)
        
        if match_ratio > threshold:
            suspicious_features.append((col, match_ratio))
            print(f"\nWarning: {col} shows suspicious relationship with target")
            print(f"Match ratio: {match_ratio:.4f}")
            print("Sample values:")
            print(pd.DataFrame({
                'feature': X[col].head(),
                'target': y.head(),
                'scaled_feature': x_scaled.head(),
                'scaled_target': y_scaled.head(),
                'difference': diff.head()
            }))
            
    return suspicious_features

def check_model_validation(model_predictions, y_true, threshold=0.95):
    """Check if model predictions are suspiciously accurate"""
    accuracy = np.mean(np.abs(model_predictions - y_true) / y_true < 0.01)
    if accuracy > threshold:
        print("\nWARNING: Model predictions are suspiciously accurate!")
        print(f"Accuracy ratio: {accuracy:.4f}")
        return False
    return True
