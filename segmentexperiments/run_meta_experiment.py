from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns  # Add seaborn import
from sklearn.base import BaseEstimator
from segmentexperiments.rapids_segment import RapidsSegmentExperiment
from segmentexperiments.neuralnet_segment import NeuralNetSegmentExperiment
from segments.meta_selector import PredictionSelector
from segmentexperiments.utils import load_segment_data  # Updated import

def verify_cuda():
    """Verify CUDA is available before running experiments"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    
    # Print GPU information
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print("\nGPU Information:")
    print(f"Device: {props.name}")
    print(f"Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {props.major}.{props.minor}\n")

def verify_dependencies():
    """Verify all required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'xgboost': 'XGBoost',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',  # Add seaborn
        'cudf': 'cuDF',
        'cuml': 'cuML',
        'lightgbm': 'LightGBM'  # Add LightGBM
    }
    
    missing = []
    versions = {}
    
    for pkg, name in required_packages.items():
        try:
            module = __import__(pkg)
            versions[name] = getattr(module, '__version__', 'unknown')
        except ImportError:
            missing.append(name)
    
    if missing:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}.\n"
            "Please install using:\n"
            "pip install " + " ".join(pkg for pkg, name in required_packages.items() if name in missing)
        )
    
    # Print installed versions
    print("\nInstalled Packages:")
    print("-" * 50)
    for name, version in versions.items():
        print(f"{name}: {version}")
    print()

def run_segment_experiments(train_data: pd.DataFrame, 
                          val_data: pd.DataFrame, 
                          test_data: pd.DataFrame,
                          output_dir: Path) -> Dict[str, pd.Series]:
    """Run all segment experiments and collect their predictions"""
    
    # Create base output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = {
        'neural': NeuralNetSegmentExperiment(output_dir / 'neural', test_mode=False),
        'rapids': RapidsSegmentExperiment(output_dir / 'rapids', test_mode=False)
    }
    
    predictions = {}
    for name, experiment in experiments.items():
        print(f"\nRunning {name} experiment...")
        experiment_dir = experiment.output_dir
        
        # Verify experiment directory exists
        if not experiment_dir.exists():
            experiment_dir.mkdir(parents=True)
        
        print(f"Starting {name} experiment with output dir: {experiment_dir}")
        experiment.run(train_data, val_data, test_data)
        
        # Update prediction paths to match new structure
        val_preds_path = experiment_dir / 'predictions' / 'validation_predictions.csv'
        test_preds_path = experiment_dir / 'predictions' / 'test_predictions.csv'
        
        print(f"Looking for prediction files in: {experiment_dir}")
        if not val_preds_path.exists() or not test_preds_path.exists():
            print("\nAvailable files:")
            for path in experiment_dir.rglob('*'):
                print(f"  {path}")
            raise FileNotFoundError(
                f"Prediction files not found for {name} experiment.\n"
                f"Expected:\n  {val_preds_path}\n  {test_preds_path}"
            )
        
        predictions[f'{name}_val'] = pd.read_csv(val_preds_path)['PremiumAmount']
        predictions[f'{name}_test'] = pd.read_csv(test_preds_path)['PremiumAmount']
        print(f"Successfully loaded predictions for {name}")
    
    return predictions

def main():
    try:
        # Verify dependencies first
        verify_dependencies()
        
        # Verify CUDA next
        verify_cuda()
        
        # Load data
        train, validate, test = load_segment_data()
        print("Data loaded successfully")
        print(f"Shapes - Train: {train.shape}, Val: {validate.shape}, Test: {test.shape}")
        
        # Setup paths
        output_dir = Path('/home/USER/Projects/ikaggle/segment_output')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {output_dir}")
        
        # Run experiments
        predictions = run_segment_experiments(train, validate, test, output_dir)
        if not predictions:
            raise RuntimeError("No predictions were generated")
            
        # Prepare meta-features
        val_predictions = pd.DataFrame({
            name.replace('_val', ''): preds 
            for name, preds in predictions.items() 
            if name.endswith('_val')
        })
        
        test_predictions = pd.DataFrame({
            name.replace('_test', ''): preds 
            for name, preds in predictions.items() 
            if name.endswith('_test')
        })
        
        # Train meta-selector
        selector = PredictionSelector(feature_importance_threshold=0.01)
        selector.train_selector(
            all_predictions=val_predictions,
            actual_values=validate['PremiumAmount'],
            feature_values=validate[['Age', 'AnnualIncome', 'HealthScore', 
                                   'CreditScore', 'NumberofDependents']]
        )
        
        # Generate meta-predictions for test set
        final_predictions = selector.predict(
            all_predictions=test_predictions,
            feature_values=test[['Age', 'AnnualIncome', 'HealthScore', 
                               'CreditScore', 'NumberofDependents']]
        )
        
        # Save final submission with actual IDs
        submission_path = selector.save_submission(
            predictions=final_predictions,
            sample_ids=test['id'],  # Add this line
            experiment_name='meta_segment'
        )
        
        print(f"\nMeta-experiment complete. Final submission saved to: {submission_path}")
    
    except Exception as e:
        print(f"Error in meta-experiment: {e}")
        raise

if __name__ == "__main__":
    main()