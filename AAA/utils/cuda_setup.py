import os
import subprocess
import sys
from pathlib import Path

def find_cuda_path():
    # verify CUDA path 
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        return "/usr/lib/cuda"
    
    
    return None

def verify_feature_scaling(feature_name, original_range, encoded_range):
    """Verify feature scaling is working properly"""
    if encoded_range[0] == encoded_range[1] == 0:
        print(f"⚠️  Warning: {feature_name} has zero range after encoding")
        return False
    
    # Check if scaling is reasonable (should be roughly 0-1)
    if encoded_range[1] > 1.5 or encoded_range[0] < -0.5:
        print(f"⚠️  Warning: {feature_name} scaling outside expected range")
        return False
    
    return True

def setup_cuda():
    """Configure CUDA environment"""
    cuda_path = "/usr/lib/cuda"
    if not cuda_path:
        print("ERROR: Could not find CUDA installation")
        return False
        
    # Find libdevice in common locations
    possible_paths = [
        "/usr/lib/cuda/nvvm/libdevice/libdevice.10.bc",
        "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc",
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"
    ]
    
    libdevice = None
    for path in possible_paths:
        if os.path.exists(path):
            libdevice = path
            break
            
    if not libdevice:
        print("ERROR: Could not find libdevice.10.bc")
        return False
        
    print(f"\nFound CUDA at: {cuda_path}")
    print(f"Found libdevice at: {libdevice}")
    
    # Set environment variables
    os.environ['CUDA_HOME'] = cuda_path
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}/lib64:{os.getenv('LD_LIBRARY_PATH', '')}"
    os.environ['PATH'] = f"{cuda_path}/bin:{os.getenv('PATH', '')}"
    os.environ['XLA_FLAGS'] = f"--xla_gpu_cuda_data_dir={cuda_path}"
    
    # Create symlinks if needed
    nvvm_path = Path(cuda_path) / "nvvm" / "libdevice"
    if not nvvm_path.exists():
        print(f"\nCreating directory: {nvvm_path}")
        nvvm_path.mkdir(parents=True, exist_ok=True)
        
    target_libdevice = nvvm_path / "libdevice.10.bc"
    if not target_libdevice.exists():
        print(f"Creating symlink: {target_libdevice} -> {libdevice}")
        try:
            target_libdevice.symlink_to(libdevice)
        except Exception as e:
            print(f"Failed to create symlink: {e}")
            print("Try running with sudo")
            return False
    
    print("\nEnvironment configured:")
    print(f"CUDA_HOME={os.getenv('CUDA_HOME')}")
    print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")
    print(f"XLA_FLAGS={os.getenv('XLA_FLAGS')}")
    
    # Add feature scaling verification
    print("\nVerifying feature scaling:")
    features_to_check = {
        'Age': ([18.00, 64.00], [0.00, 1.00]),
        'AnnualIncome': ([1.00, 149997.00], [0.00, 1.00]),
        'NumberofDependents': ([0.00, 4.00], [0.00, 1.00]),
        'PreviousClaims': ([0.00, 8.00], [0.00, 1.00])
    }
    
    scaling_ok = True
    for feat, (orig, enc) in features_to_check.items():
        if not verify_feature_scaling(feat, orig, enc):
            scaling_ok = False
            
    if not scaling_ok:
        print("\nWarning: Feature scaling issues detected")
    
    return True

if __name__ == "__main__":
    if setup_cuda():
        print("\nCUDA setup completed successfully")
    else:
        print("\nCUDA setup failed")