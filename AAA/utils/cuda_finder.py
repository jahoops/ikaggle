import os
import logging
import sys
import subprocess
import glob
from typing import Dict, List
from pathlib import Path
from ikaggle.AAA.config.paths import (
    CUDA_PATHS, CACHE_PATHS, SYSTEM_PATHS, CUDA_BIN_PATHS  # Add CUDA_BIN_PATHS import
)

class CUDAFinder:
    """Find CUDA installation and required files"""
    
    @staticmethod
    def verify_cuda_binaries():
        """Verify CUDA binaries are available"""
        required_binaries = ['ptxas', 'nvcc']
        missing = []
        
        for binary in required_binaries:
            binary_path = CUDA_PATHS.get(binary)
            if not binary_path or not os.path.exists(binary_path):
                # Try finding in system path
                try:
                    from shutil import which
                    if not which(binary):
                        missing.append(binary)
                except:
                    missing.append(binary)
                    
        return len(missing) == 0, missing

    @staticmethod
    def find_ptxas():
        """Find ptxas binary in system"""
        try:
            # Try direct paths first
            direct_paths = [
                "/usr/local/cuda/bin/ptxas",
                "/usr/local/cuda-12.1/bin/ptxas",
                "/usr/bin/ptxas",
                CUDA_PATHS['ptxas']
            ]
            
            for path in direct_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    return path
                    
            # Try using which
            from shutil import which
            ptxas_path = which('ptxas')
            if ptxas_path:
                return ptxas_path
        except Exception as e:
            print(f"Error finding ptxas: {e}")
        return None

    @staticmethod
    def setup_cuda_env():
        """Setup CUDA environment variables"""
        # Setup environment
        env_vars = {
            'CUDA_HOME': CUDA_PATHS['lib'],
            'CUDA_PATH': CUDA_PATHS['lib'],
            'CUDA_ROOT': CUDA_PATHS['lib'],
            'CUDA_DIR': os.path.dirname(CUDA_PATHS['libdevice']),
            'LD_LIBRARY_PATH': f"{CUDA_PATHS['lib']}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            'PATH': f"{os.path.dirname(CUDA_PATHS['ptxas'])}:{CUDA_PATHS['bin']}:{os.environ.get('PATH', '')}",
            'XLA_FLAGS': f"--xla_gpu_cuda_data_dir={os.path.dirname(CUDA_PATHS['libdevice'])}",
            'TF_XLA_FLAGS': "--tf_xla_cpu_global_jit --tf_xla_auto_jit=2"
        }
        
        # Update environment
        os.environ.update(env_vars)
        
        print("\nCUDA environment configured:")
        print(f"CUDA_HOME: {env_vars['CUDA_HOME']}")
        print(f"PATH includes ptxas dir: {os.path.dirname(CUDA_PATHS['ptxas'])}")
        print(f"Using libdevice at: {CUDA_PATHS['libdevice']}")
        
        return True

def find_cuda_installations() -> Dict[str, List[str]]:
    """Find all CUDA-related installations and paths on the system"""
    cuda_info = {
        'cuda_paths': [],
        'nvcc_paths': [],
        'libcudart_paths': [],
        'conda_cuda': [],
        'environment_vars': {}
    }
    
    # Check common locations
    common_paths = [
        '/usr/lib/nvidia-cuda-toolkit',
        os.path.expanduser('~/.local/cuda'),
        os.path.expanduser('~/miniconda3/envs')
    ]
    
    # Find CUDA installations
    for base_path in common_paths:
        if os.path.exists(base_path):
            # Look for CUDA version directories
            cuda_dirs = glob.glob(f"{base_path}/cuda*") + [base_path]
            for cuda_dir in cuda_dirs:
                if os.path.exists(os.path.join(cuda_dir, 'bin/nvcc')):
                    cuda_info['cuda_paths'].append(cuda_dir)
    
    # Find nvcc compilers
    try:
        which_nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
        cuda_info['nvcc_paths'].append(which_nvcc)
    except subprocess.CalledProcessError:
        pass
    
    # Find libcudart.so files
    for lib_path in ['/usr/lib', '/usr/lib64', '/usr/local/cuda/lib64']:
        if os.path.exists(lib_path):
            libcudart_files = glob.glob(f"{lib_path}/libcudart.so*")
            cuda_info['libcudart_paths'].extend(libcudart_files)
    
    # Check Conda environments
    conda_path = os.path.expanduser('~/miniconda3/envs')
    if os.path.exists(conda_path):
        for env in os.listdir(conda_path):
            cuda_path = os.path.join(conda_path, env, 'lib/libcudart.so')
            if os.path.exists(cuda_path):
                cuda_info['conda_cuda'].append(cuda_path)
    
    # Get CUDA-related environment variables
    for var in ['CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT', 'LD_LIBRARY_PATH']:
        if var in os.environ:
            cuda_info['environment_vars'][var] = os.environ[var]
    
    return cuda_info

def print_cuda_info():
    """Print detailed CUDA installation information"""
    cuda_info = find_cuda_installations()
    
    print("\nCUDA Installation Information:")
    print("=" * 50)
    
    print("\nCUDA Paths:")
    for path in cuda_info['cuda_paths']:
        print(f"  - {path}")
    
    print("\nNVCC Compiler Paths:")
    for path in cuda_info['nvcc_paths']:
        print(f"  - {path}")
    
    print("\nCUDA Runtime Libraries:")
    for path in cuda_info['libcudart_paths']:
        print(f"  - {path}")
    
    print("\nConda CUDA Installations:")
    for path in cuda_info['conda_cuda']:
        print(f"  - {path}")
    
    print("\nCUDA Environment Variables:")
    for var, value in cuda_info['environment_vars'].items():
        print(f"  {var}: {value}")
    
    # Try to determine primary CUDA installation
    primary_cuda = None
    if cuda_info['nvcc_paths']:
        primary_cuda = os.path.dirname(os.path.dirname(cuda_info['nvcc_paths'][0]))
    elif cuda_info['cuda_paths']:
        primary_cuda = cuda_info['cuda_paths'][0]
    
    if primary_cuda:
        print(f"\nPrimary CUDA installation appears to be: {primary_cuda}")
    else:
        print("\nWarning: Could not determine primary CUDA installation")

if __name__ == "__main__":
    print_cuda_info()
