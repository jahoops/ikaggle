import numpy as np
import tensorflow as tf
import gc
import os
import psutil
import matplotlib.pyplot as plt

def print_memory_usage():
    """Print current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")

def check_tensor_stats(tensor, name="tensor"):
    """Check statistics of a tensor for debugging"""
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()
    
    # Check for NaN and Inf
    nan_count = np.isnan(tensor).sum()
    inf_count = np.isinf(tensor).sum()
    
    # Basic statistics
    tensor_min = np.min(tensor) if nan_count < tensor.size else "N/A"
    tensor_max = np.max(tensor) if nan_count < tensor.size else "N/A"
    tensor_mean = np.mean(tensor) if nan_count < tensor.size else "N/A"
    tensor_std = np.std(tensor) if nan_count < tensor.size else "N/A"
    
    print(f"Stats for {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  Min: {tensor_min}")
    print(f"  Max: {tensor_max}")
    print(f"  Mean: {tensor_mean}")
    print(f"  Std: {tensor_std}")

def plot_gradient_flow(model, inputs, targets):
    """Plot gradient flow in a model for debugging"""
    # Record gradients
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = model.loss(targets, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Plot gradients
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(gradients)), 
            [tf.reduce_mean(tf.abs(g)) if g is not None else 0 for g in gradients])
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/USER/Projects/ikaggle/KISS/gradient_flow.png')
    plt.close()

def fix_numerical_instability(model, zero_values=True):
    """Fix potential numerical instability in model weights"""
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            # Clip weights
            layer.kernel.assign(tf.clip_by_value(layer.kernel, -1.0, 1.0))
        if hasattr(layer, 'bias') and layer.bias is not None and zero_values:
            # Zero out very small values (potential underflows)
            small_mask = tf.abs(layer.bias) < 1e-6
            layer.bias.assign(tf.where(small_mask, tf.zeros_like(layer.bias), layer.bias))

def enable_mixed_precision():
    """Enable mixed precision training for better numerical stability"""
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

def disable_mixed_precision():
    """Disable mixed precision training"""
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision disabled")

def clean_memory():
    """Clean memory to avoid OOM errors"""
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Specify GPU memory to be released
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.reset_memory_stats(gpu)
            print("GPU memory reset")
        except:
            print("Could not reset GPU memory")
            
    print_memory_usage()
