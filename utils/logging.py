import logging
import os

def setup_logging():
    """Configure logging with file and console handlers"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("experiment_logs.log"),
            logging.StreamHandler()
        ]
    )
    
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
