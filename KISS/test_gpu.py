import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Devices:", tf.config.list_physical_devices('GPU'))

if tf.config.list_physical_devices('GPU'):
    # Run a simple operation on GPU to verify it works
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c.numpy())
        print("GPU test PASSED!")
else:
    print("No GPU detected. GPU test FAILED!")
