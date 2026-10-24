import tensorflow as tf
import torch
import subprocess
import sys

def check_nvidia_smi():
    """Check if nvidia-smi can detect any GPUs"""
    try:
        nvidia_output = subprocess.check_output(['nvidia-smi']).decode()
        print("=== NVIDIA-SMI Output ===")
        print(nvidia_output)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not found or failed. GPU may not be properly installed.")

def check_tensorflow():
    """Check TensorFlow GPU configuration"""
    print("\n=== TensorFlow Configuration ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print("\nVisible devices:")
    print(tf.config.list_physical_devices())
    
    # Test GPU availability
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("\nGPU Test calculation successful")
    except RuntimeError as e:
        print(f"\nGPU Test failed: {e}")

def check_pytorch():
    """Check PyTorch GPU configuration"""
    print("\n=== PyTorch Configuration ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

def main():
    """Run all GPU checks"""
    print("Running GPU diagnostics...\n")
    check_nvidia_smi()
    check_tensorflow()
    check_pytorch()

if __name__ == "__main__":
    main()
