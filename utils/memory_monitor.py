# /mnt/c/home/USER/ensemble/utils/memory_monitor.py
import psutil
import logging

def setup_memory_logging(log_file='memory_usage.log'):
    """Set up logging configuration for memory monitoring."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def log_memory_usage(tag=""):
    """Log current memory usage with an optional tag."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
    message = f"Memory Usage{' (' + tag + ')' if tag else ''}: {memory_mb:.2f} MB"
    logging.info(message)