import pandas as pd
import numpy as np

def validate_dataset(df, target_col='PremiumAmount'):
    """Validate dataset for potential issues"""
    issues = []
    
    # Check for target-related columns
    target_keywords = ['premium', 'amount', 'payment', 'price', 'cost']
    suspicious_cols = [
        col for col in df.columns 
        if any(keyword in col.lower() for keyword in target_keywords)
    ]
    
    if suspicious_cols:
        issues.append(f"Found columns that may contain target info: {suspicious_cols}")
    
    # Check for ID columns that might encode target
    id_cols = [col for col in df.columns if 'id' in col.lower()]
    for col in id_cols:
        if df[col].corr(df[target_col]) > 0.1:
            issues.append(f"ID column {col} correlates with target!")
            
    return issues
