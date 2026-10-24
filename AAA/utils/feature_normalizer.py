import numpy as np
from collections import defaultdict

def normalize_features(feature_matrix, feature_names, verbose=True):
    if feature_matrix is None or feature_names is None:
        raise ValueError("Input matrix and feature names cannot be None")
        
    normalized = feature_matrix.copy().astype(float)
    stats = defaultdict(dict)
    
    try:
        for i, fname in enumerate(feature_names):
            col = normalized[:, i]
            if not np.any(col):
                continue
                
            if verbose:
                print(f"\nProcessing {fname}:")
                print(f"Original values: {col}")
                
            if 'Income' in fname:
                # Obtain unique income values and their sorted order
                unique_values = np.unique(col)
                if verbose:
                    print(f"Unique income values: {unique_values}")
                # Build a mapping: each unique value -> its rank (as a float)
                value_to_rank = {float(val): idx for idx, val in enumerate(np.sort(unique_values))}
                if verbose:
                    print(f"Value to rank mapping: {value_to_rank}")
                n_unique = len(unique_values)
                if n_unique > 1:
                    mapped = np.array([
                        -0.4 + (0.8 * value_to_rank[float(val)] / (n_unique - 1)) for val in col
                    ])
                else:
                    mapped = np.zeros_like(col)
                normalized[:, i] = mapped
                if verbose:
                    print(f"Mapped values: {normalized[:, i]}")
                    print(f"Differences: {np.diff(normalized[:, i])}")
            elif 'Age' in fname:
                # Standardize age features to smaller range
                col = (col - np.mean(col)) / (np.std(col) + 1e-8)
                col = np.clip(col, -1, 1) * 0.25
                normalized[:, i] = col
            else:
                # General normalization
                median = np.median(col[col != 0])
                iqr = np.percentile(col[col != 0], 75) - np.percentile(col[col != 0], 25)
                if iqr == 0:
                    iqr = np.abs(median) if median != 0 else 1.0
                    
                col = (col - median) / (iqr + 1e-8)
                
                # Scale based on feature type
                if any(x in fname for x in ['_add_', '_max_', '_min_']):
                    col = np.clip(col, -1.5, 1.5) * 0.3
                else:
                    col = np.clip(col, -2, 2) * 0.4
                
                normalized[:, i] = col
            
            # Store stats
            stats[fname] = {
                'range': [float(np.min(normalized[:, i])), float(np.max(normalized[:, i]))],
                'std': float(np.std(normalized[:, i]))
            }
            
        # Final validation
        if np.any(np.abs(normalized) > 1.0):
            normalized = np.clip(normalized, -1.0, 1.0)
            
        if verbose:
            print("\nFeature normalization stats:")
            for fname, stat in stats.items():
                if np.any(normalized[:, feature_names.index(fname)] != 0):
                    print(f"  {fname}: range [{stat['range'][0]:.2f}, {stat['range'][1]:.2f}], "
                          f"std: {stat['std']:.2f}")
                    
        return normalized
        
    except Exception as e:
        print(f"Normalization error: {str(e)}")
        return None
