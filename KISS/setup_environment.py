import subprocess
import sys
import os

def run_command(command):
    """Run a shell command and print output"""
    print(f"\n> Running: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result.returncode == 0

def install_requirements():
    """Install all required packages for the insurance premium project"""
    print("===== Installing Required Packages =====")
    
    # Core data science packages
    packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "scipy"
    ]
    
    # Machine learning packages
    ml_packages = [
        "tensorflow",  # Base TensorFlow
        "keras",       # High-level API
    ]
    
    # Utility packages
    utility_packages = [
        "tqdm",           # Progress bars
        "joblib",         # Parallelism
        "ipykernel",      # For Jupyter integration
        "ipython",        # Enhanced Python shell
        "notebook"        # Jupyter notebook
    ]
    
    # Check if we're in a conda environment
    in_conda = os.environ.get('CONDA_DEFAULT_ENV') is not None
    
    if in_conda:
        print(f"Installing in conda environment: {os.environ.get('CONDA_DEFAULT_ENV')}")
        
        # Install core packages with conda
        if not run_command(f"conda install -y {' '.join(packages)}"):
            # Fallback to pip if conda fails
            run_command(f"{sys.executable} -m pip install {' '.join(packages)}")
            
        # For TensorFlow, we might need more specific installations
        run_command(f"conda install -y tensorflow")
        
        # Install other utilities
        run_command(f"conda install -y {' '.join(utility_packages)}")
    else:
        print("Installing with pip (no conda environment detected)")
        # Install everything with pip
        run_command(f"{sys.executable} -m pip install {' '.join(packages + ml_packages + utility_packages)}")
    
    # Special handling for GPU dependencies
    try:
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            print("\n===== GPU NOT DETECTED =====")
            print("Installing TensorFlow GPU dependencies...")
            if in_conda:
                print("For conda, try installing:")
                print("conda install -c conda-forge cudatoolkit=11.8.0 cuda-nvcc=11.8.89")
                print("conda install -c nvidia cudnn=8.6.0.163")
                print("\nAlternatively, install the TensorFlow GPU pip package:")
                print("pip install tensorflow[and-cuda]")
            else:
                print("For pip, installing tensorflow with CUDA:")
                run_command(f"{sys.executable} -m pip install tensorflow[and-cuda]")
        else:
            print("\n===== GPU SUCCESSFULLY DETECTED =====")
            print(f"Found {len(tf.config.list_physical_devices('GPU'))} GPU(s)")
            print(tf.config.list_physical_devices('GPU'))
    except ImportError:
        print("TensorFlow not accessible. Install with:")
        print("conda install -y tensorflow")
        print("or")
        print("pip install tensorflow")

def verify_installation():
    """Verify that all required packages are installed correctly"""
    print("\n===== Verifying Installations =====")
    
    packages_to_check = [
        "pandas", "numpy", "matplotlib", "seaborn", 
        "sklearn", "tensorflow", "scipy"
    ]
    
    for package in packages_to_check:
        try:
            module = __import__(package)
            print(f"✓ {package} installed successfully (version: {module.__version__})")
        except (ImportError, AttributeError):
            print(f"× {package} NOT installed or version not available")

def create_test_script():
    """Create a test script that verifies imports and GPU access"""
    test_script = """
# filepath: /home/USER/Projects/ikaggle/KISS/verify_libraries.py
import sys
print(f"Python version: {sys.version}")

# Core data science libraries
import pandas as pd
print(f"pandas version: {pd.__version__}")

import numpy as np
print(f"numpy version: {np.__version__}")

from matplotlib import pyplot as plt
print(f"matplotlib version: {plt.__version__}")

import seaborn as sns
print(f"seaborn version: {sns.__version__}")

# Machine learning libraries
from sklearn import __version__ as sklearn_version
print(f"scikit-learn version: {sklearn_version}")

import tensorflow as tf
print(f"tensorflow version: {tf.__version__}")

# Check for GPU
print("\\nGPU Information:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU is available: Found {len(gpus)} GPU(s)")
    # Try a simple GPU operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
        c = tf.matmul(a, b)
        print("GPU computation test:", c.numpy())
else:
    print("No GPU available")
    
# Test all imports used in the project
print("\\nTesting imports for project-specific modules:")
try:
    from utils.data_processor import InsuranceDataProcessor
    print("✓ InsuranceDataProcessor")
except ImportError as e:
    print(f"× InsuranceDataProcessor: {e}")
    
try:
    from utils.segmentation import InsuranceSegmenter
    print("✓ InsuranceSegmenter")
except ImportError as e:
    print(f"× InsuranceSegmenter: {e}")
    
try:
    from model.group_trainer import GroupSpecialistTrainer
    print("✓ GroupSpecialistTrainer")
except ImportError as e:
    print(f"× GroupSpecialistTrainer: {e}")
    
# Print system information
import platform
print("\\nSystem Information:")
print(f"OS: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
"""
    
    # Write the test script
    with open("/home/USER/Projects/ikaggle/KISS/verify_libraries.py", "w") as f:
        f.write(test_script)
    
    print("Created verification script at: /home/USER/Projects/ikaggle/KISS/verify_libraries.py")
    print("Run it with: python /home/USER/Projects/ikaggle/KISS/verify_libraries.py")

if __name__ == "__main__":
    install_requirements()
    verify_installation()
    create_test_script()
    print("\nSetup complete! Run the verification script to test your environment.")
