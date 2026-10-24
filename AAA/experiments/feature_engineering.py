import numpy as np
import pandas as pd
from ikaggle.AAA.utils.feature_combiner import FeatureCombiner
from ikaggle.AAA.config.feature_config import FEATURE_CONFIGS, get_gene_length, decode_gene, compute_derived_feature

def debug_print(*args, verbose=False, **kwargs):
    """Helper function for conditional printing"""
    if verbose:
        print(*args, **kwargs)

def decode_individual(individual, X_full):
    """Convert binary individual to feature values with combinations"""
    feature_values = {}
    current_pos = 0
    num_features = len(X_full.columns)
    
    # Decode base features
    for feature in X_full.columns:
        # Check activation bit
        is_active = bool(individual[current_pos])
        current_pos += 1
        
        if FEATURE_CONFIGS[feature]['type'] == 'variable':
            gene_length = get_gene_length(feature)
            gene = individual[current_pos:current_pos + gene_length]
            # Fix empty binary string issue
            if gene and any(gene):  # Check if gene exists and has any 1s
                value = decode_gene(gene, feature) if is_active else 0
            else:
                value = 0
            feature_values[feature] = value
            current_pos += gene_length
        else:
            feature_values[feature] = is_active
    
    # Compute derived features
    for feature in X_full.columns:
        if FEATURE_CONFIGS[feature]['type'] == 'derived':
            feature_values[feature] = compute_derived_feature(feature, feature_values)
    
    # Decode first order combinations (pairs)
    num_first_order = 200
    combo_start = current_pos
    for i in range(num_first_order):
        pos = combo_start + i * 5
        if pos + 21 >= len(individual):  # 5 genes * 4 bits + activation
            break
            
        if individual[pos]:  # If combination is active
            try:
                # Get feature indices (using 8 bits each for more range)
                f1_bits = individual[pos+1:pos+9]
                f2_bits = individual[pos+9:pos+17]
                f1_idx = int(''.join(map(str, f1_bits)), 2) % num_features
                f2_idx = int(''.join(map(str, f2_bits)), 2) % num_features
                
                # Get operation and features
                op_bits = individual[pos+17:pos+20]
                operation = FeatureCombiner.get_operation_code(op_bits)
                use_raw = individual[pos+20]
                
                f1_name = X_full.columns[f1_idx]
                f2_name = X_full.columns[f2_idx]
                
 
                f1_values = feature_values.get(f1_name, 0)
                f2_values = feature_values.get(f2_name, 0)
                
                combo_name = f"Pair_{i}_{f1_name}_{operation}_{f2_name}"
                feature_values[combo_name] = FeatureCombiner.combine_features(
                    f1_values, f2_values, operation)
                    
            except Exception as e:
                print(f"Warning: Skipping pair combination {i}: {str(e)}")
                continue
    
    # Update current position
    current_pos = combo_start + num_first_order * 5
    
    # Decode third order combinations (triplets)
    num_third_order = 50
    for i in range(num_third_order):
        pos = current_pos + i * 7  # 7 genes per triplet
        if pos + 31 >= len(individual):  # 7 genes * 4 bits + activation
            break
            
        if individual[pos]:  # If combination is active
            try:
                # Get three feature indices
                f1_bits = individual[pos+1:pos+9]
                f2_bits = individual[pos+9:pos+17]
                f3_bits = individual[pos+17:pos+25]
                
                f1_idx = int(''.join(map(str, f1_bits)), 2) % num_features
                f2_idx = int(''.join(map(str, f2_bits)), 2) % num_features
                f3_idx = int(''.join(map(str, f3_bits)), 2) % num_features
                
                # Get operations
                op1_bits = individual[pos+25:pos+28]
                op2_bits = individual[pos+28:pos+31]
                
                operation1 = FeatureCombiner.get_operation_code(op1_bits)
                operation2 = FeatureCombiner.get_operation_code(op2_bits)
                use_raw = individual[pos+31]
                
                # Get feature names
                f1_name = X_full.columns[f1_idx]
                f2_name = X_full.columns[f2_idx]
                f3_name = X_full.columns[f3_idx]
                
                if use_raw:
                    f1_values = feature_values.get(f1_name, 0)
                    f2_values = feature_values.get(f2_name, 0)
                    f3_values = feature_values.get(f3_name, 0)
                else:
                    f1_values = X_full[:, f1_idx]
                    f2_values = X_full[:, f2_idx]
                    f3_values = X_full[:, f3_idx]
                
                # Combine features in sequence: (f1 op1 f2) op2 f3
                temp_result = FeatureCombiner.combine_features(f1_values, f2_values, operation1)
                combo_name = f"Triple_{i}_{f1_name}_{operation1}_{f2_name}_{operation2}_{f3_name}"
                feature_values[combo_name] = FeatureCombiner.combine_features(
                    temp_result, f3_values, operation2)
                    
            except Exception as e:
                print(f"Warning: Skipping triplet combination {i}: {str(e)}")
                continue
    
    return feature_values

def encode_features(individual, X_full, verbose=False):
    """Convert binary genome to actual features"""
    try:
        # Convert input to DataFrame if needed
        if isinstance(X_full, tuple):
            X_full = pd.DataFrame(X_full[0])  # Assuming first element contains data
        elif not isinstance(X_full, pd.DataFrame):
            X_full = pd.DataFrame(X_full)
            
        if verbose:
            print(f"\nEncoding features for data shape: {X_full.shape}")
            print(f"Data type: {type(X_full)}")
            print(f"Columns: {X_full.columns.tolist()}")
        
        features = []
        current_pos = 0
        
        # Get genome from individual or use individual if it's already a list
        genome = individual.genome if hasattr(individual, 'genome') else individual
        
        # Process base features first
        base_features = {}
        for col in X_full.columns:
            # Check activation bit
            if current_pos >= len(genome):
                print(f"WARNING: Genome position {current_pos} exceeds genome length {len(genome)}")
                break
                
            if genome[current_pos]:
                base_features[col] = X_full[col].values
                if verbose:
                    print(f"Activated feature: {col}")
                    
            current_pos += 1
            
            # Handle variable length features
            if col in FEATURE_CONFIGS and FEATURE_CONFIGS[col]['type'] == 'variable':
                gene_length = get_gene_length(col)
                current_pos += gene_length
        
        if not base_features:
            print("WARNING: No base features were activated!")
            # Activate at least one feature to prevent complete failure
            first_col = X_full.columns[0]
            base_features[first_col] = X_full[first_col].values
            print(f"Automatically activated {first_col}")
        
        # Convert dictionary to list of feature arrays
        features = list(base_features.values())
        
        # Convert to numpy array
        if features:
            features_array = np.column_stack(features)
            if verbose:
                print(f"Final feature array shape: {features_array.shape}")
            return features_array
        else:
            print("ERROR: No features were encoded")
            return None
            
    except Exception as e:
        print(f"Error in encode_features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def normalize_features(feature_matrix, feature_names):
    """Normalize features"""
    # ...existing normalize_features code...
