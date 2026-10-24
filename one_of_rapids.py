import ctypes
import os
import sys
from typing import Dict, List

def check_environment() -> Dict[str, str]:
    """Verify conda environment configuration"""
    env_info = {
        "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None'),
        "python_exec": sys.executable,
        "python_version": sys.version.split()[0]
    }
    
    for key, value in env_info.items():
        print(f"{key}: {value}")
    return env_info

def test_cuinit():
    """Test CUDA initialization"""
    cuda = ctypes.CDLL('libcuda.so')
    result = cuda.cuInit(0)
    if result == 0:
        print("CUDA initialization: successful")
    else:
        print(f"CUDA initialization failed: error {result}")

if __name__ == "__main__":
    check_environment()
    print("\nTesting CUDA:")
    test_cuinit()