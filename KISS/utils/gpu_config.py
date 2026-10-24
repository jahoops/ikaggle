"""
GPU configuration utilities to address TensorFlow numerical precision issues.
"""
import os
import tensorflow as tf
import numpy as np

def configure_gpu_for_stability():
    """
    Configure GPU settings for numerical stability by disabling problematic optimizations
    and setting appropriate precision policies.
    
    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    try:
        # Check if GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("No GPU devices available - using CPU")
            return False
            
        print(f"Found {len(gpus)} GPU(s)")
        
        # Configure memory growth to avoid OOM errors
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Enabled memory growth for {gpu}")
            except:
                print(f"Failed to set memory growth for {gpu}")
                
        # Set appropriate precision policy
        # Use full float32 instead of mixed precision
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Disable XLA JIT compilation which can cause precision issues
        tf.config.optimizer.set_jit(False)
        
        # Set environment variables to force certain TF behaviors
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_CPU_ALLOCATOR_LIMIT_MB'] = '0'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Make operations deterministic (slightly slower)
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        print("GPU configured for numerical stability")
        return True
        
    except Exception as e:
        print(f"Error configuring GPU: {str(e)}")
        return False

def get_distribution_strategy():
    """
    Get an appropriate distribution strategy based on available hardware.
    
    Returns:
        tf.distribute.Strategy: A TensorFlow distribution strategy
    """
    try:
        # Check if GPU is available
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            print("Using OneDeviceStrategy with GPU")
        else:
            strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
            print("Using OneDeviceStrategy with CPU")
        return strategy
    except Exception as e:
        print(f"Error creating distribution strategy: {str(e)}")
        print("Falling back to default strategy")
        return tf.distribute.get_strategy()

def create_model_on_device(build_fn, device='CPU'):
    """
    Create a model on a specific device to avoid device placement issues.
    
    Args:
        build_fn: Function that builds and returns a model
        device: Device to build the model on ('CPU' or 'GPU')
        
    Returns:
        The model created on the specified device
    """
    with tf.device(f'/{device}:0'):
        model = build_fn()
    return model
