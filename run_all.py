import argparse
from pathlib import Path
from experiments.models import MODEL_REGISTRY
from utils import config
import logging
import os

logger = logging.getLogger(__name__)

def get_active_venv():
    """Detect active virtual environment"""
    return os.environ.get('VIRTUAL_ENV', '').split('/')[-1]

def is_tf_environment():
    """Check if current environment supports TensorFlow"""
    return get_active_venv() == 'vnext'

def get_compatible_experiments():
    """Get list of experiments that can run in current environment"""
    base_experiments = [
        #'catboost',
        #'knn',
        #'lasso',
        #'lightgbm', 
        #'linear', 
        #'neuralnet',
        #'randomforest',
        #'ridge', 
        #'svr',
        'xgboost'
    ]

    # OVERRIDE for testing
    # base_experiments = ['linear','ridge', 'randomforest', 'svr']    
    return base_experiments

def run_experiment(model_type, test_mode):
    output_base = Path(config.get('experiment', 'output_path'))
    output_dir = output_base / model_type.capitalize()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ExperimentClass = MODEL_REGISTRY[model_type]
    experiment = ExperimentClass(output_dir=output_dir, test_mode=test_mode)
    experiment.run()

def main():
    parser = argparse.ArgumentParser(description="Run all ML experiments.")
    parser.add_argument('--test', action='store_true', help='Run in test mode with minimal settings.')
    args = parser.parse_args()
    test_mode = args.test

    current_env = get_active_venv()
    logger.info(f"Running in environment: {current_env}")
    
    experiments = get_compatible_experiments()
    logger.info(f"Will run experiments: {experiments}")
    
    failed_models = []
    for model_type in experiments:
        logger.info(f"Running experiment for model: {model_type}")
        try:
            run_experiment(model_type, test_mode)
            logger.info(f"Successfully completed: {model_type}")
        except Exception as e:
            logger.error(f"Failed to run {model_type}: {str(e)}")
            failed_models.append(model_type)
            continue
    
    if failed_models:
        logger.warning(f"The following models failed: {failed_models}")
    logger.info("Finished all experiments")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()