import os
import tensorflow as tf
import subprocess
from pathlib import Path

def find_cuda_root():
    """Find CUDA installation root"""
    # Add debug output
    print("\nSearching for CUDA installation...")
    
    cuda_paths = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CONDA_PREFIX"),  # Add conda environment
        "/home/USER/miniconda3/envs/ikaggle-gpu",  # Add specific conda env
        "/usr/lib/cuda",
        "/home/USER/.local/cuda/nvvm/libdevice/",
        "/home/USER/miniconda3/pkgs/torchtriton-3.1.0-py310/lib/python3.10/site-packages/triton/backends/nvidia/bin/",
        

    ]
    
    # Try nvidia-smi first to detect CUDA version
    try:
        cuda_version = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader']).decode().strip()
        print(f"Found NVIDIA driver version: {cuda_version}")
    except:
        print("Warning: nvidia-smi not found or failed")
    
    # Check each path
    for path in cuda_paths:
        if not path:
            continue
            
        path = str(Path(path).resolve())  # Resolve symlinks
        print(f"Checking path: {path}")
        
        # Check for cuda binary directory
        if os.path.exists(os.path.join(path, "bin/nvcc")):
            print(f"Found CUDA installation at: {path}")
            return path
        
        # Check lib directory
        lib_path = os.path.join(path, "lib64")
        if os.path.exists(lib_path):
            cuda_libs = [f for f in os.listdir(lib_path) if "libcudart.so" in f]
            if cuda_libs:
                print(f"Found CUDA libraries at: {lib_path}")
                return os.path.dirname(lib_path)
    
    # Try finding cuda in system
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc']).decode().strip()
        if nvcc_path:
            cuda_root = str(Path(nvcc_path).parent.parent)
            print(f"Found CUDA via nvcc at: {cuda_root}")
            return cuda_root
    except:
        pass
    
    return None

def setup_cuda_env():
    """Setup CUDA environment variables"""
    # Hard-code the CUDA path here (replace with your actual path):
    cuda_root = "/home/USER/miniconda3/envs/ikaggle-gpu"
    
    if not cuda_root:
        raise RuntimeError("Could not find CUDA installation")
    
    # Get conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    
    # Setup paths
    lib_paths = [
        f"{cuda_root}/lib64",
        f"{cuda_root}/extras/CUPTI/lib64",
        f"{conda_prefix}/lib" if conda_prefix else None
    ]
    lib_paths = [p for p in lib_paths if p]  # Remove None values
    
    # Setup environment variables
    env_vars = {
        'CUDA_HOME': cuda_root,
        'CUDA_ROOT': cuda_root,
        'CUDA_PATH': cuda_root,
        'PATH': f"{cuda_root}/bin:{os.environ.get('PATH', '')}",
        'LD_LIBRARY_PATH': ':'.join(lib_paths + [os.environ.get('LD_LIBRARY_PATH', '')]),
        'XLA_FLAGS': f"--xla_gpu_cuda_data_dir={cuda_root}",
        'CUDA_DIR': cuda_root,
        'TF_CUDA_PATHS': cuda_root
    }
    
    # Update environment
    os.environ.update(env_vars)
    
    # Print configuration
    print("\nCUDA Configuration:")
    print(f"CUDA_HOME: {cuda_root}")
    print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    print(f"PATH: {os.environ['PATH']}")
    
    # Verify CUDA setup
    try:
        subprocess.check_call(['nvcc', '--version'])
    except:
        print("Warning: nvcc not found in PATH")

def configure_gpu():
    """Configure GPU with proper settings"""
    try:
        setup_cuda_env()
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            return False
            
        # Enable memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            
        # Enable XLA
        tf.config.optimizer.set_jit(True)
        
        return True
    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False

def configure_xla():
    """Configure XLA for better performance."""
    tf.config.optimizer.set_jit(True)

def get_gpu_config(compute_capability=(8, 9)):
    """Get GPU configuration for model training"""
    return {
        'batch_size': 256,
        'dense_layers': [512, 256, 128],
        'learning_rate': 0.001,
        'mixed_precision': True,
        'jit_compile': True
    }

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
