import yaml
import joblib
import pandas as pd
from pathlib import Path
from preprocessing import preprocess_data
from model import train_insurance_model, rmsle, validate_gpu, get_optimal_workers
from datetime import datetime
import xgboost as xgb
import sys
import numpy as np
import multiprocessing
import argparse

# Get project root directory (BBB folder) when run from Projects
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_gpu_availability():
    """Check if GPU is available for XGBoost and LightGBM"""
    try:
        # Include dummy labels when creating the DMatrix
        dummy_data = np.random.randn(10, 10)
        dummy_labels = np.random.randn(10)
        xgb_gpu = xgb.train(
            {'tree_method': 'hist', 'device': 'cuda'},
            xgb.DMatrix(dummy_data, label=dummy_labels)
        )
        print("XGBoost GPU support: Available")
    except Exception as e:
        print(f"XGBoost GPU support: Not available ({str(e)})")

    try:
        # Check LightGBM GPU availability
        import lightgbm as lgb
        lgb_gpu = lgb.Dataset(np.random.randn(10, 10), label=np.random.randn(10)).construct()
        print("LightGBM GPU support: Available")
    except Exception as e:
        print(f"LightGBM GPU support: Not available ({str(e)})")

def create_test_data(real_data, sample_size=1000):
    """Create a small test dataset that maintains the same structure"""
    if isinstance(real_data, pd.DataFrame):
        return real_data.sample(n=min(sample_size, len(real_data)), random_state=42)
    elif isinstance(real_data, np.ndarray):
        indices = np.random.RandomState(42).choice(len(real_data), min(sample_size, len(real_data)), replace=False)
        return real_data[indices]
    return real_data

def main():  
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Train insurance model')
    parser.add_argument('--test', action='store_true', help='Run in test mode with small dataset')
    parser.add_argument('--quick', action='store_true', help='Run with minimal iterations for testing')
    args = parser.parse_args()
    
    if args.test:
        print("\nRunning in TEST mode with reduced dataset")
    if args.quick:
        print("\nRunning in QUICK mode with minimal iterations")
    
    print(f"Project root: {PROJECT_ROOT}")
    start_time = datetime.now()
    print(f"Starting training at {start_time}")
    
    try:
        # Load config
        config_path = PROJECT_ROOT / 'config' / 'params.yaml'
        print(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Modify config for quick mode
        if args.quick:
            config['model']['params']['n_estimators'] = 10
            config['model']['gbm_params']['n_estimators'] = 10
            config['model']['xgboost']['n_estimators'] = 10
            config['model']['lightgbm']['num_boost_round'] = 10
        
        # Load and read data first
        train_path = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
        test_path = PROJECT_ROOT / 'data' / 'raw' / 'test.csv'
        
        print(f"\nLoading data from:")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
        
        train_df = pd.read_csv(str(train_path))  # Convert Path to string
        test_df = pd.read_csv(str(test_path))    # Convert Path to string
        
        # Reset indices before any processing
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Reduce dataset size for testing
        if args.test:
            print("\nReducing dataset size for testing...")
            train_df = create_test_data(train_df, sample_size=1000)
            test_df = create_test_data(test_df, sample_size=1000)
            print(f"Test data shapes - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Preprocess data with DataFrames
        X_train, X_test, y_train, scaler, feature_cols = preprocess_data(
            train_df,
            test_df,
            config['preprocessing']
        )
        
        if args.test:
            print("\nTest mode: Verifying data shapes:")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train shape: {y_train.shape}")
        
        print("\nTraining model with all enhancements...")
        # Update to pass feature_cols to train_insurance_model
        model, y_pred, cv_rmsle, feature_mask = train_insurance_model(
            X_train, X_test, y_train,
            config['model'],
            feature_cols
        )
        
        # Create submission with mode indicator in filename
        mode_suffix = '_test' if args.test else '_quick' if args.quick else ''
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        submission_path = PROJECT_ROOT / 'data' / 'processed' / f'submission_rmsle_{cv_rmsle:.4f}{mode_suffix}_{timestamp}.csv'
        
        submission = pd.DataFrame({
            'id': test_df['id'],
            'Premium Amount': y_pred
        })
        submission.to_csv(submission_path, index=False)
        
        # Save model artifacts
        if not args.test:  # Don't save test mode models
            artifacts_path = PROJECT_ROOT / 'models' / f'model_rmsle_{cv_rmsle:.4f}_{timestamp}'
            artifacts_path.mkdir(exist_ok=True)
            
            joblib.dump(model, artifacts_path / 'ensemble_model.pkl')
            joblib.dump(scaler, artifacts_path / 'scaler.pkl')
            joblib.dump(feature_mask, artifacts_path / 'feature_mask.pkl')
        
        # Print summary
        print("\n" + "="*50)
        print("Training Summary:")
        print(f"Mode: {'TEST' if args.test else 'QUICK' if args.quick else 'FULL'}")
        print(f"Cross-validation RMSLE: {cv_rmsle:.4f}")
        print(f"Training duration: {datetime.now() - start_time}")
        print(f"Submission saved to: {submission_path}")
        if not args.test:
            print(f"Model artifacts saved to: {artifacts_path}")
        print("="*50)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Current working directory: {Path.cwd()}")
        raise

if __name__ == "__main__":
    main()
