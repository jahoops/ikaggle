import os
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.data_manager import get_data, features
from utils.segmentation import InsuranceSegmenter
from model.group_trainer import GroupSpecialistTrainer
from model.super_nn import SuperNN
from model.preprocess import engineer_features
import glob

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    print("Starting group-based model training...")
    
    # Create directories
    results_dir = '/home/USER/Projects/ikaggle/KISS/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Step 1: Load and preprocess data
    print("\nLoading data...")
    train_data = get_data('just_train')
    validate_data = get_data('validate')
    contest_data = get_data('contest')
    
    # Engineer features
    train_data_engineered, _ = engineer_features(train_data.copy())
    validate_data_engineered, _ = engineer_features(validate_data.copy())
    contest_data_engineered, _ = engineer_features(contest_data.copy())
    
    # Extract target
    y_train = train_data['PremiumAmount']
    y_validate = validate_data['PremiumAmount']
    
    # Log transform the target
    y_train_log = np.log1p(y_train)
    y_validate_log = np.log1p(y_validate)
    
    # Step 2: Perform feature importance analysis and segmentation
    print("\nAnalyzing feature importance and creating segments...")
    segmenter = InsuranceSegmenter()
    
    # Analyze feature importance using different methods
    segmenter.analyze_feature_importance(train_data_engineered, y_train, method='random_forest')
    segmenter.analyze_feature_importance(train_data_engineered, y_train, method='correlation')
    
    # Create segments based on top features
    groups = segmenter.create_segments(train_data_engineered, y_train, method='kmeans', n_clusters=4)
    
    # Step 3: Evaluate existing models by group
    print("\nEvaluating existing model performance by group...")
    group_trainer = GroupSpecialistTrainer()
    
    # Assign groups to all datasets
    group_assignments = group_trainer.assign_groups(segmenter, train_data_engineered, validate_data_engineered)
    
    # Find existing submission files
    submission_files = glob.glob(f"{results_dir}/submission_model_*.csv")
    if len(submission_files) == 0:
        print("No existing submission files found. Train models first using regular ensemble.")
        return
        
    print(f"Found {len(submission_files)} submission files.")
    
    # Evaluate models by group
    performance = group_trainer.evaluate_models_by_group(
        submission_files,
        group_assignments['test'],
        y_validate
    )
    
    # Step 4: Load existing models for fine-tuning
    print("\nLoading models for specialization...")
    general_models = []
    for i in range(7):  # Try to load all 7 standard models
        try:
            model = SuperNN.load_model(i)
            general_models.append(model)
            print(f"Loaded model {i}")
        except:
            print(f"Could not load model {i}")
    
    if len(general_models) == 0:
        print("No existing models found. Train models first using regular ensemble.")
        return
    
    # Step 5: Train specialized models for each group
    from model.preprocess import preprocess_data  # Import preprocess function
    
    # Prepare data for training
    X_train_scaled = preprocess_data(train_data_engineered)
    X_validate_scaled = preprocess_data(validate_data_engineered)
    X_contest_scaled = preprocess_data(contest_data_engineered)
    
    # Train specialized models
    specialized_models = group_trainer.train_specialized_models(
        general_models,
        X_train_scaled,
        y_train_log,
        X_validate_scaled,
        y_validate_log
    )
    
    # Step 6: Create final group-based ensemble
    print("\nCreating final group-based ensemble...")
    ensemble_path = group_trainer.create_group_based_ensemble