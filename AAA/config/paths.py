import os
import glob
import pandas as pd

# Base directories based on actual system configuration
CONDA_ENV = "/home/USER/miniconda3/envs/ikaggle-gpu"
SYSTEM_CUDA = "/usr/lib/nvidia-cuda-toolkit"
NVCC_PATH = "/usr/bin/nvcc"

# User-writable CUDA directories
USER_HOME = os.path.expanduser('~')
USER_CUDA_DIR = os.path.join(USER_HOME, '.local/cuda')
USER_CUDA_CACHE = os.path.join(USER_HOME, '.cache/cuda')
USER_CUDA_NVVM = os.path.join(USER_CUDA_DIR, 'nvvm/libdevice')  # Add this line

# Create required directories
for path in [USER_CUDA_DIR, USER_CUDA_CACHE, USER_CUDA_NVVM]:
    os.makedirs(path, exist_ok=True)

def find_cuda_conda_pkg():
    """Find CUDA package in conda environment"""
    conda_pkgs = glob.glob("/home/USER/miniconda3/pkgs/cuda-*")
    return sorted(conda_pkgs)[-1] if conda_pkgs else None

CUDA_CONDA_PKG = find_cuda_conda_pkg()

# Update CUDA paths based on discovered locations
CUDA_PATHS = {
    'ptxas': os.path.join(SYSTEM_CUDA, 'bin/ptxas'),  # System CUDA
    'nvcc': NVCC_PATH,  # System NVCC
    'libdevice': os.path.join(CONDA_ENV, 'lib/libdevice.10.bc'),  # Conda environment
    'lib': os.path.join(CONDA_ENV, 'lib'),  # Conda environment libs
    'cuda_home': CONDA_ENV,  # Use Conda environment as CUDA home
    'system_cuda': SYSTEM_CUDA  # System CUDA installation
}

# Add CUDA binary paths
CUDA_BIN_PATHS = [
    CUDA_PATHS['nvcc'],
    os.path.join(CONDA_ENV, 'bin'),
    '/usr/local/cuda/bin',
    '/usr/bin'
]
CUDA_BIN_PATHS = [p for p in CUDA_BIN_PATHS if os.path.exists(p)]

# System paths including both system and conda CUDA
SYSTEM_PATHS = [
    CUDA_PATHS['lib'],
    os.path.join(SYSTEM_CUDA, 'lib64'),
    os.path.join(SYSTEM_CUDA, 'lib'),
    '/usr/local/cuda/lib64',
    '/usr/lib/x86_64-linux-gnu'
]

# Cache paths - consolidated definition
CACHE_PATHS = {
    'cuda': USER_CUDA_CACHE,
    'models': '/home/USER/Projects/ikaggle/AAA/models',
    'temp': '/home/USER/Projects/ikaggle/AAA/temp',
    'checkpoints': '/home/USER/Projects/ikaggle/AAA/models/checkpoints',
    'generations': '/home/USER/Projects/ikaggle/AAA/models/generations'
}

# Create all cache directories
for path in CACHE_PATHS.values():
    os.makedirs(path, exist_ok=True)

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Data paths - Updated to use correct base path and filenames
DATA_PATHS = {
    'train': '/home/USER/Projects/ikaggle/AAA/data/cleaned_train.csv',
    'test': '/home/USER/Projects/ikaggle/AAA/data/cleaned_test.csv',
    'validate': '/home/USER/Projects/ikaggle/AAA/data/cleaned_validate.csv',
    'testing': '/home/USER/Projects/ikaggle/AAA/data/cleaned_testing.csv'
}

# Add path verification
def verify_data_paths():
    """Verify data paths and print detailed status"""
    missing_files = []
    for key, path in DATA_PATHS.items():
        exists = os.path.exists(path)
        print(f"Data file '{key}': {path}")
        print(f"  Exists: {exists}")
        if exists:
            df = pd.read_csv(path)
            print(f"  Columns: {list(df.columns)}")
        else:
            missing_files.append(key)
            
    if missing_files:
        print(f"\nWarning: Missing data files: {missing_files}")
        print("Please ensure all required data files are present.")
        return False
    return True

# Create symlinks for required files if they don't exist
def setup_cuda_paths():
    """Setup CUDA paths and symlinks"""
    # Ensure NVVM directory exists
    nvvm_dir = os.path.join(CUDA_PATHS['cuda_home'], 'nvvm', 'libdevice')
    os.makedirs(nvvm_dir, exist_ok=True)
    
    # Create symlink for libdevice if needed
    libdevice_target = os.path.join(nvvm_dir, 'libdevice.10.bc')
    if not os.path.exists(libdevice_target) and os.path.exists(CUDA_PATHS['libdevice']):
        try:
            os.symlink(CUDA_PATHS['libdevice'], libdevice_target)
        except OSError as e:
            print(f"Warning: Could not create symlink: {e}")

# Update environment on import
os.environ.update({
    'CUDA_HOME': CUDA_PATHS['cuda_home'],
    'LD_LIBRARY_PATH': ':'.join([p for p in SYSTEM_PATHS if os.path.exists(p)]),
    'PATH': f"{os.path.dirname(CUDA_PATHS['nvcc'])}:{os.environ.get('PATH', '')}"
})

# Call setup on import
setup_cuda_paths()

# Call verification on import
if not verify_data_paths():
    print("Data path verification failed!")
