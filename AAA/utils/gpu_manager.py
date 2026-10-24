import tensorflow as tf
import os
import contextlib
import subprocess
import shutil
import logging
from ikaggle.AAA.utils.cuda_finder import CUDAFinder

class GPUManager:
    def __init__(self):
        # Initialize logging first
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize basic variables
        self.gpu_available = False
        self.initialized = False
        self.compute_capability = (8, 9)  # RTX 4060 Ti
        self.device_memory = None
        self.device = None
        
        # Initialize GPU in correct order
        self._configure_environment()
        self._initialize_gpu()
    
    def _configure_environment(self):
        """Configure CUDA environment"""
        try:
            os.environ.update({
                'CUDA_VISIBLE_DEVICES': '0',
                'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
                'TF_CPP_MIN_LOG_LEVEL': '1',
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'TF_CUDA_COMPUTE_CAPABILITIES': '8.9',
                'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
                'TF_XLA_FLAGS': '--tf_xla_enable_xla_devices',
                'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
            })
            
            # Create cache directories
            cache_dir = os.path.expanduser('~/.cache/cuda')
            os.makedirs(cache_dir, exist_ok=True)
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            raise
    
    def _initialize_gpu(self):
        """Initialize GPU and create context"""
        try:
            # Reset any existing GPU configuration
            tf.keras.backend.clear_session()
            
            physical_devices = tf.config.list_physical_devices('GPU')
            if not physical_devices:
                self.logger.warning("No GPU devices found")
                return
                
            # Configure first GPU
            self.device = physical_devices[0]
            tf.config.experimental.set_memory_growth(self.device, True)
            
            # Create and test context
            with tf.device('/GPU:0'):
                # Force context creation
                tf.random.uniform([1], dtype=tf.float32)
                
            self.gpu_available = True
            self.logger.info(f"GPU initialized successfully with compute capability {self.compute_capability}")
            
        except Exception as e:
            self.logger.error(f"GPU initialization failed: {e}")
            self.gpu_available = False
    
    def get_device_context(self):
        """Get device context for model operations"""
        if self.gpu_available:
            return tf.device('/GPU:0')
        return contextlib.nullcontext()
    
    def _find_libdevice(self):
        """Find libdevice.10.bc file"""
        # First check known locations
        for path in self.libdevice_paths:
            if os.path.exists(path):
                return path
                
        # Search system if not found in known locations
        try:
            found = subprocess.check_output(['find', '/usr', '-name', 'libdevice.10.bc']).decode().strip()
            if found:
                return found.split('\n')[0]  # Take first match
        except subprocess.CalledProcessError:
            pass
        return None
        
    def _get_gpu_memory(self):
        """Get available GPU memory"""
        try:
            nvidia_output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader']).decode()
            return int(nvidia_output.strip()) * 0.8  # Use 80% of available memory
        except:
            return 8 * 1024  # Default to 8GB if can't determine
    
    def _detect_gpu_capabilities(self):
        """Detect GPU capabilities for optimization"""
        try:
            device = tf.config.list_physical_devices('GPU')[0]
            props = tf.config.experimental.get_device_details(device)
            self.compute_capability = props.get('compute_capability', (8, 9))  # Default to 8.9 for RTX 4060 Ti
            self.max_shared_memory = props.get('shared_memory_size_per_multiprocessor', 64 * 1024)
            
            # Configure stream priorities
            if self.gpu_available:
                for device in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(device, False)
                    tf.config.experimental.set_virtual_device_configuration(
                        device,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=self._get_gpu_memory(),
                            priority=0  # Set default priority
                        )]
                    )
        except Exception as e:
            print(f"GPU capability detection error: {e}")
            self.compute_capability = (8, 9)
            self.max_shared_memory = 64 * 1024
    
    def initialize(self):
        """Initialize GPU configuration"""
        if self.initialized:
            return self.gpu_available

        try:
            # Clear any existing GPU memory
            tf.keras.backend.clear_session()
            
            physical_devices = tf.config.list_physical_devices('GPU')
            if not physical_devices:
                return False

            gpu = physical_devices[0]
            
            # Configure GPU memory
            memory_limit = self._get_gpu_memory()
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit))]
            )
            
            # Create logical devices after memory config
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(memory_limit))]
            )
            
            # Test GPU with a small computation
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0]])
                b = tf.matmul(a, a)
            
            self.gpu_available = True
            self.gpu_initialized = True
            
        except Exception as e:
            print(f"GPU initialization error: {e}")
            self.gpu_available = False
            
        self.initialized = True
        return self.gpu_available
    
    @contextlib.contextmanager
    def device_context(self):
        """Context manager for GPU operations"""
        if not self.initialized:
            self.initialize()
            
        try:
            if self.gpu_available:
                # Ensure clean GPU state
                tf.keras.backend.clear_session()
                with tf.device('/GPU:0'):
                    yield
            else:
                with tf.device('/CPU:0'):
                    yield
        finally:
            tf.keras.backend.clear_session()
    
    def get_recommended_batch_size(self):
        """Get optimized batch size based on GPU memory and compute capability"""
        if not self.gpu_available:
            return 64
            
        memory_mb = self._get_gpu_memory() / 1024
        cc_major, cc_minor = self.compute_capability
        
        # Optimize batch size based on compute capability
        if cc_major >= 8:  # Ampere and newer
            if memory_mb > 12000:
                return 512
            elif memory_mb > 8000:
                return 384
            else:
                return 256
        else:  # Older architectures
            if memory_mb > 12000:
                return 256
            elif memory_mb > 8000:
                return 192
            else:
                return 128
    
    def get_optimized_model_config(self):
        """Get optimized model configuration based on GPU capabilities"""
        return {
            'use_mixed_precision': True,
            'enable_xla': True,
            'enable_tensorrt': False,  # TensorRT optimization not needed for this workload
            'batch_size': self.get_recommended_batch_size(),
            'threads': 1,  # Single thread per GPU for better performance
            'memory_growth': False,  # Pre-allocate memory
            'compute_capability': self.compute_capability
        }

def setup_gpu():
    """Setup GPU with proper memory growth and logging"""
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if not physical_devices:
            print("No GPU devices found. Running on CPU.")
            return None, False
            
        print(f"Found {len(physical_devices)} GPU(s)")
        gpu = physical_devices[0]
        
        # Get total GPU memory
        total_memory = 16380  # Your GPU's total memory in MiB
        target_memory = int(total_memory * 0.95)  # Use 95% of available memory
        
        # Configure memory growth and limit
        tf.config.experimental.set_memory_growth(gpu, False)  # Disable memory growth
        tf.config.set_logical_device_configuration(
            gpu,
            [tf.config.LogicalDeviceConfiguration(memory_limit=target_memory)]
        )
        
        # Test GPU with larger batch
        with tf.device('/GPU:0'):
            # Test with larger matrix
            a = tf.random.normal([1000, 1000])
            b = tf.matmul(a, a)
            print("GPU test successful!")
            print(f"Using device: {b.device}")
            print(f"Allocated GPU memory: {target_memory}MiB")
            
        return gpu, True
        
    except Exception as e:
        print(f"GPU setup failed: {str(e)}")
        return None, False
