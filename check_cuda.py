import torch
import sys
import subprocess
from packaging import version

def check_version_compatibility(pytorch_cuda_version, rapids_version):
    """Check compatibility between PyTorch CUDA and RAPIDS versions"""
    compatibility_matrix = {
        '11.8': ['23.12', '23.10', '23.08'],
        '12.1': ['24.02']  # Future compatibility
    }
    
    pytorch_base = '.'.join(pytorch_cuda_version.split('.')[:2])
    rapids_base = '.'.join(rapids_version.split('.')[:2])
    
    compatible_versions = compatibility_matrix.get(pytorch_base, [])
    return rapids_base in compatible_versions

def check_cuda_installation():
    print("Environment Details:")
    print("-" * 50)
    
    # Check required packages
    required_packages = {
        'torch': torch,
        'sklearn': None,
        'numpy': None
    }
    
    for package in required_packages:
        try:
            module = __import__(package)
            print(f"{package} version: {module.__version__}")
        except ImportError:
            print(f"{package} not installed")
    
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        pytorch_cuda_version = torch.version.cuda
        rapids_version = None
        
        print("\nGPU Details:")
        print("-" * 50)
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"PyTorch CUDA version: {pytorch_cuda_version}")
        
        # Add GPU memory info
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached GPU memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        try:
            # Get CUDA version from nvidia-smi
            nvidia_smi = subprocess.check_output(['nvidia-smi']).decode()
            print("\nNVIDIA-SMI Output:")
            print("-" * 50)
            print(nvidia_smi)
            
            # Check for RAPIDS
            try:
                import cudf
                import cuml
                rapids_version = cudf.__version__
                print("\nRAPIDS Details:")
                print("-" * 50)
                print(f"cuDF version: {cudf.__version__}")
                print(f"cuML version: {cuml.__version__}")
                
                # Check version compatibility
                if check_version_compatibility(pytorch_cuda_version, rapids_version):
                    print("PyTorch CUDA and RAPIDS versions are compatible.")
                else:
                    print("Warning: PyTorch CUDA and RAPIDS versions may not be compatible.")
            except ImportError as e:
                print(f"\nRAPIDS not installed: {str(e)}")
                print("To install RAPIDS, visit: https://rapids.ai/start.html")
            
        except Exception as e:
            print(f"\nError running nvidia-smi: {e}")
    
    print("\nCUDA Build Configuration:")
    print("-" * 50)
    print(f"CUDA build enabled: {torch.backends.cuda.is_built()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

if __name__ == "__main__":
    check_cuda_installation()