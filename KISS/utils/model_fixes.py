"""
Utility functions to fix model prediction issues
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_predictions(predictions, original=None, title="Prediction Distribution"):
    """
    Diagnose prediction issues by creating visualizations and statistics
    """
    print(f"\nDiagnostics for {title}:")
    print(f"Shape: {predictions.shape}")
    print(f"Min: {np.min(predictions)}, Max: {np.max(predictions)}")
    print(f"Mean: {np.mean(predictions)}, Median: {np.median(predictions)}")
    print(f"Zeros: {np.sum(predictions == 0)}, NaNs: {np.sum(np.isnan(predictions))}")
    print(f"Values at min cap (50.0): {np.sum(predictions == 50.0)}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50)
    plt.title(f"{title} Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"/home/USER/Projects/ikaggle/KISS/results/diagnostics_{title.replace(' ', '_').lower()}.png")
    
    # If original is provided, compare
    if original is not None:
        plt.figure(figsize=(10, 6))
        plt.scatter(original, predictions, alpha=0.5)
        plt.title(f"Original vs {title}")
        plt.xlabel("Original")
        plt.ylabel(title)
        plt.savefig(f"/home/USER/Projects/ikaggle/KISS/results/comparison_{title.replace(' ', '_').lower()}.png")
    
    return {
        'min': np.min(predictions),
        'max': np.max(predictions),
        'mean': np.mean(predictions),
        'median': np.median(predictions),
        'zeros': np.sum(predictions == 0),
        'nans': np.sum(np.isnan(predictions)),
        'at_min_cap': np.sum(predictions == 50.0)
    }

def fix_model_initialization(input_dim, model_type='normal'):
    """
    Create models with fixed initialization to avoid prediction issues
    """
    # Set global random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Use a more balanced initialization for stability
    kernel_initializer = tf.keras.initializers.HeNormal(seed=42)
    
    # Use mixed precision for stability
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create model
    if model_type == 'normal':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,),
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear',
                                kernel_initializer=kernel_initializer)
        ])
    elif model_type == 'wide':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,),
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu',
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear',
                                kernel_initializer=kernel_initializer)
        ])
    else:  # deep
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,),
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu',
                                kernel_initializer=kernel_initializer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear',
                                kernel_initializer=kernel_initializer)
        ])
    
    # Use a mix of mean absolute error (more robust) and mean squared error
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber_loss',  # More robust to outliers
        metrics=['mae', 'mse']
    )
    
    return model
