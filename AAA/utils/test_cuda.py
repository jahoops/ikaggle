import os
import tensorflow as tf
import subprocess
from pathlib import Path

def test_cuda_setup():
    """Validate CUDA configuration"""
    print("Testing CUDA Setup:")
    print("-" * 50)

    # 1. Check CUDA paths
    cuda_home = os.getenv('CUDA_HOME')
    print(f"CUDA_HOME: {cuda_home}")
    if not Path(cuda_home).exists():
        print("❌ CUDA_HOME path does not exist")
        return False

    # 2. Check libdevice
    libdevice = Path(cuda_home) / "nvvm" / "libdevice" / "libdevice.10.bc"
    print(f"Checking libdevice: {libdevice}")
    if not libdevice.exists():
        print("❌ libdevice.10.bc not found")
        return False

    # 3. Check CUDA version compatibility
    try:
        nvidia_out = subprocess.check_output(["nvidia-smi"]).decode()
        cuda_driver = nvidia_out.split("CUDA Version:")[1].split()[0]
        print(f"CUDA Driver Version: {cuda_driver}")
        
        tf_cuda = tf.sysconfig.get_build_info()["cuda_version"]
        print(f"TensorFlow CUDA Version: {tf_cuda}")
        
        if not tf_cuda.startswith(cuda_driver.split('.')[0]):
            print("⚠️ Warning: TensorFlow CUDA version mismatch")
    except:
        print("❌ Could not verify CUDA versions")

    # 4. Test GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("❌ No GPUs found by TensorFlow")
        return False
    
    print(f"✅ Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.device_type}: {gpu.name}")

    # 5. Quick GPU test
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("✅ GPU matrix multiplication test passed")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    if test_cuda_setup():
        print("\n✅ CUDA setup validation successful")
    else:
        print("\n❌ CUDA setup validation failed")
