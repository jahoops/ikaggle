
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
print("\nGPU Information:")
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
print("\nTesting imports for project-specific modules:")
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
print("\nSystem Information:")
print(f"OS: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"Machine: {platform.machine()}")
print(f"Processor: {platform.processor()}")
