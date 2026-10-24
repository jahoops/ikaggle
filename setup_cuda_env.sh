#!/bin/bash

# Find CUDA installation
CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
if [ -z "$CUDA_ROOT" ]; then
    echo "CUDA installation not found. Installing CUDA toolkit..."
    sudo apt-get update
    sudo apt-get install -y nvidia-cuda-toolkit
    CUDA_ROOT=/usr/local/cuda
fi

# CUDA paths
export CUDA_HOME=$CUDA_ROOT
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Create symbolic link for libdevice
LIBDEVICE_PATH="$CUDA_HOME/nvvm/libdevice"
if [ -f "$LIBDEVICE_PATH/libdevice.10.bc" ]; then
    echo "Found libdevice at: $LIBDEVICE_PATH"
    # Create link in current directory
    ln -sf "$LIBDEVICE_PATH/libdevice.10.bc" ./libdevice.10.bc
else
    echo "libdevice not found at expected location. Searching for it..."
    FOUND_LIBDEVICE=$(find /usr -name "libdevice.10.bc" 2>/dev/null)
    if [ ! -z "$FOUND_LIBDEVICE" ]; then
        echo "Found libdevice at: $FOUND_LIBDEVICE"
        ln -sf "$FOUND_LIBDEVICE" ./libdevice.10.bc
    else
        echo "Error: Could not find libdevice.10.bc"
        exit 1
    fi
fi

# XLA configuration - point to the found libdevice location
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"

# Additional CUDA configurations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Verify CUDA installation
echo "Checking CUDA installation..."
nvcc --version
nvidia-smi

# Activate conda environment
conda activate ikaggle-gpu

# Test TensorFlow GPU
python -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs found: {gpus}')
if gpus:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.matmul(a, a)
        print('GPU test successful')
"
