import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

# Test GPU
if tf.config.list_physical_devices('GPU'):
    print("\nTesting GPU computation...")
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("GPU test successful!")
else:
    print("\nNo GPU found. Check TensorFlow installation.")
    print("Make sure CUDA and cuDNN are properly installed.")
