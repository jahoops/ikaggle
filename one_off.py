import os
import tensorflow as tf

# Limit TensorFlow to use only the first GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set memory growth to prevent TensorFlow from allocating all GPU memory
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

print(f"TensorFlow Version: {tf.__version__}")
print(f"Has 'register_fft_plugin': {hasattr(tf.experimental, 'register_fft_plugin')}")

gpu_count = len(tf.config.list_physical_devices('GPU'))
print(f"GPU Count: {gpu_count}")

import numba.cuda
print(numba.cuda.gpus)