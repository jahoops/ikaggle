# Your Goal: The objectives of this challenge are to predict insurance premiums based on various factors.
# Submissions are evaluated using the Root Mean Squared Logarithmic Error (RMSLE).
# For each id row in the test set, you must predict the continuous target PremiumAmount. The file should contain a header and have the following format:

# id,PremiumAmount
# 1200000,1102.545
# 1200001,1102.545
# 1200002,1102.545
# etc.

# Input data is in the input folder called train.csv and test.csv

# Output goes to the output folder

# Check the script that cleans the data: preprocess.py
# Define different feature combinations to test: output/cleaned_train.csv
# Set up a list of models to experiment with
# Create a function to evaluate models
# Implement an automated experimentation loop
# Save results to a comparison table
# Always keep the best model of each type for later ensembling
# Create submission using the best models in generate_submission.py
# First run

# python auto_experiment.py --version v1

# After crash, continue the same version
# python auto_experiment.py --version v1

# Start fresh
# python auto_experiment.py --version v1 --reset



import logging
import os

# Disable oneDNN optimizations to suppress related warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow and CUDA warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Import TensorFlow after setting environment variables
import tensorflow as tf

import argparse
from pathlib import Path
from datetime import datetime
from experiments.experiment_random_forest import run_random_forest
from experiments.experiment_xgboost import run_xgboost
from experiments.experiment_svr import run_svr
from experiments.experiment_linear import run_linear
from experiments.experiment_lightgbm import run_lightgbm
from experiments.experiment_catboost import run_catboost
from experiments.experiment_knn import run_knn  # Updated import
from experiments.experiment_neural_net import run_neural_net  # Updated import
from .utils import manage_resources, check_resources, generate_performance_report  # Changed to relative import
from .utils import save_state, load_state  # Changed to relative import
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_output_dir(version):
    """Create versioned output directory"""
    output_dir = Path('output') / version
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

def run_experiments(version='v1'):
    """Run all experiments and manage their states."""
    # Step 1: Run data preprocessing
    # ...existing code...
    
    output_dir = setup_output_dir(version)
    experiments = [ 
        ("XGBoost", run_xgboost),
        ("SVR", run_svr),
        ("Linear", run_linear),
        ("LightGBM", run_lightgbm),
        ("CatBoost", run_catboost),
        ("KNN", run_knn),
        ("NeuralNet", run_neural_net),
        ("RandomForest", run_random_forest)
    ]

    # Load previous state if exists
    state = load_state(output_dir)
    completed_experiments = state['completed']  # List of dicts
    failed_experiments = state['failed']
    crashed_experiments = set(state['crashed'])
    
    # Extract set of completed model names for checking
    completed_model_names = set(exp['name'] for exp in completed_experiments)
    
    logger.info(f"Loaded previous state - Completed: {len(completed_experiments)}, Failed: {len(failed_experiments)}, Crashed: {len(crashed_experiments)}")
    
    results = []
    best_models = {}
    
    for name, experiment_fn in experiments:
        # Skip if already completed or previously crashed
        if name in completed_model_names:
            logger.info(f"Skipping {name} - already completed")
            continue
        if name in crashed_experiments:
            logger.info(f"Skipping {name} - previously crashed")
            continue
            
        try:
            resources_before = check_resources()
            logger.info(f"Resources before {name}: {resources_before}")
            
            with manage_resources():
                logger.info(f"Starting experiment: {name}")
                model_info = experiment_fn(output_dir)
                results.append(model_info)
                best_models[name] = model_info
                
                if model_info['status'] == 'success':
                    # Instead of adding just the name, append the entire model_info dict
                    completed_experments.append({
                        'name': model_info['name'],
                        'score': model_info['score']
                    })
                    completed_model_names.add(name)  # Update the set
                else:
                    failed_experiments.append((name, model_info['error']))
                    crashed_experiments.add(name)
                
                save_state(output_dir, 
                          completed_experiments,  # Pass list of dicts
                          failed_experiments,
                          list(crashed_experiments))
            
            resources_after = check_resources()
            logger.info(f"Resources after {name}: {resources_after}")
            
        except Exception as e:
            error_msg = f"{name} experiment failed: {str(e)}"
            logger.error(error_msg)
            failed_experiments.append((name, str(e)))
            crashed_experiments.add(name)  # Mark as crashed for future runs
            
            # Save state after crash
            save_state(output_dir,
                      completed_experiments,  # Pass list of dicts
                      failed_experiments,
                      list(crashed_experiments))
            
            with manage_resources():
                pass
            continue

    # Log experiment summary
    logger.info("\n=== Experiment Summary ===")
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {len(results)}")
    logger.info(f"Failed: {len(failed_experiments)}")
    
    if failed_experiments:
        logger.info("\nFailed experiments:")
        for name, error in failed_experiments:
            logger.info(f"- {name}: {error}")
    
    # Save report even if some experiments failed
    try:
        report = generate_performance_report(results, best_models)
        with open(output_dir / 'model_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # Save failed experiments log
        with open(output_dir / 'failed_experiments.json', 'w') as f:
            json.dump(failed_experiments, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving reports: {str(e)}", exc_info=True)
    
    return report, failed_experiments

def get_default_version():
    """Generate default version using timestamp"""
    return datetime.now().strftime('v%Y%m%d_%H%M%S')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default=get_default_version(),
                      help='Version tag for this experiment run (default: timestamp)')
    parser.add_argument('--reset', action='store_true',
                      help='Reset experiment state and start fresh')
    args = parser.parse_args()
    
    logger.info(f"Starting experiments version: {args.version}")
    
    # Handle reset flag
    if args.reset:
        state_file = Path('output') / args.version / 'experiment_state.json'
        if state_file exists():
            state_file.unlink()
            logger.info("Reset: Cleared previous experiment state")
    
    run_experiments(version=args.version)

