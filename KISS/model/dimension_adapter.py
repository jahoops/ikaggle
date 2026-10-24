import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda

class InputDimensionAdapter:
    """
    Utility class to handle input dimension mismatches between models and data
    """
    @staticmethod
    def adapt_features(features, target_dim):
        """Adapt feature array to match target dimensions"""
        if features is None:
            return np.zeros((0, target_dim))
            
        current_dim = features.shape[1]
        
        if current_dim == target_dim:
            return features
        
        if current_dim > target_dim:
            # Too many features - select subset
            print(f"Reducing feature dimensions from {current_dim} to {target_dim}")
            return features[:, :target_dim]
        else:
            # Too few features - pad with zeros
            print(f"Expanding feature dimensions from {current_dim} to {target_dim}")
            padding = np.zeros((features.shape[0], target_dim - current_dim))
            return np.hstack([features, padding])
    
    @staticmethod
    def create_adapter_model(model, input_dims):
        """
        Create an adapter wrapper around a model to handle dimension mismatches
        
        Args:
            model: The trained Keras model
            input_dims: Dictionary with original and combined dims
                {'original': 41, 'combined': 8}
        
        Returns:
            A new model that adapts inputs to match the original model's expectations
        """
        # Get expected dimensions from model - FIXED VERSION
        try:
            # Try first approach: get from input layer output shape (recommended)
            expected_original_dim = model.get_layer('input_original').output_shape[1]
            expected_combined_dim = model.get_layer('input_combined').output_shape[1]
        except (AttributeError, ValueError, IndexError):
            try:
                # Try second approach: get from model inputs directly
                expected_original_dim = model.inputs[0].shape[1]
                expected_combined_dim = model.inputs[1].shape[1]
            except (IndexError, AttributeError):
                # Last resort: use provided dimensions
                print("Warning: Could not determine input shapes from model, using provided dimensions")
                expected_original_dim = input_dims.get('original', 20)  # Default to 20 if not provided
                expected_combined_dim = input_dims.get('combined', 8)   # Default to 8 if not provided
        
        # Create new inputs with actual dimensions
        new_input_original = Input(shape=(input_dims['original'],), name='new_input_original')
        new_input_combined = Input(shape=(input_dims['combined'],), name='new_input_combined')
        
        # Create adapters
        if input_dims['original'] == expected_original_dim:
            adapted_original = new_input_original
        elif input_dims['original'] > expected_original_dim:
            # Select first N features
            adapted_original = Lambda(
                lambda x: x[:, :expected_original_dim],
                name='original_reducer'
            )(new_input_original)
        else:
            # Pad with zeros
            padding = Input(shape=(expected_original_dim - input_dims['original'],),
                          tensor=tf.zeros((tf.shape(new_input_original)[0], 
                                         expected_original_dim - input_dims['original'])))
            adapted_original = Concatenate(name='original_expander')([new_input_original, padding])
            
        # Same for combined features
        if input_dims['combined'] == expected_combined_dim:
            adapted_combined = new_input_combined
        elif input_dims['combined'] > expected_combined_dim:
            adapted_combined = Lambda(
                lambda x: x[:, :expected_combined_dim],
                name='combined_reducer'
            )(new_input_combined)
        else:
            padding = Input(shape=(expected_combined_dim - input_dims['combined'],),
                          tensor=tf.zeros((tf.shape(new_input_combined)[0],
                                         expected_combined_dim - input_dims['combined'])))
            adapted_combined = Concatenate(name='combined_expander')([new_input_combined, padding])
        
        # Feed adapted inputs to original model
        output = model([adapted_original, adapted_combined])
        
        # Create new model
        adapter_model = tf.keras.Model(
            inputs=[new_input_original, new_input_combined],
            outputs=output,
            name='dimension_adapted_model'
        )
        
        return adapter_model

def adapt_inputs(X, expected_shapes):
    """Adapt input tensors to match expected shapes"""
    adapted_inputs = []
    
    for i, (X_input, expected_shape) in enumerate(zip(X, expected_shapes)):
        current_shape = X_input.shape[1]
        
        if current_shape != expected_shape:
            print(f"Adapting input {i}: {current_shape} → {expected_shape}")
            
            if current_shape > expected_shape:
                # Reduce dimensions
                adapted_inputs.append(X_input[:, :expected_shape])
            else:
                # Expand with zeros
                padding = np.zeros((X_input.shape[0], expected_shape - current_shape))
                adapted_inputs.append(np.hstack([X_input, padding]))
        else:
            adapted_inputs.append(X_input)
            
    return adapted_inputs
