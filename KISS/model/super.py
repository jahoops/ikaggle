import os
import time
import pandas as pd
import numpy as np
from itertools import combinations
from numpy import inf
from utils.data_manager import get_data, features, target
from model.super_nn import SuperNN, ActuarialLayer, RMSLELoss, RMSLEMetric
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from utils.math import safe_rmsle
import tensorflow as tf
import random
from collections import deque
from model.preprocess import engineer_features

# Import the ensemble functionality
from .ensemble import ensemble_submissions, get_optimal_weights

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Check TensorFlow version
print(f"Using TensorFlow version: {tf.__version__}")

# Configure TensorFlow to allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth set for GPU: {gpu}")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPUs found.")

def load_and_shuffle_combinations(file_path, top_n=40):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Skip the header line
        lines = lines[1:]
        # Skip blank lines
        lines = [line.strip() for line in lines if line.strip()]
        # Split each line into feature pair and RMSLE score
        lines = [line.rsplit(',', 1) for line in lines]
        # Convert feature pair to tuple and score to float
        lines = [(eval(line[0]), float(line[1])) for line in lines]
    # Take the top N combinations
    top_combinations = lines[:top_n]
    # Shuffle the combinations
    random.shuffle(top_combinations)
    return deque(top_combinations)

def preprocess_data(train_df, validate_df, features):
    """
    Preprocess both training and validation data.
    Returns scaled original features and targets for both datasets.
    """
    # Handle missing values (simple imputation with mean)
    train_df = train_df.fillna(train_df.mean())
    validate_df = validate_df.fillna(train_df.mean())  # Use training mean for validation data
    
    # Check for invalid values
    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()
    validate_df = validate_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Store original features separately
    X_train_original = train_df[features].copy()
    X_validate_original = validate_df[features].copy()

    # Separate target (using imported target variable)
    y_train = train_df[target]
    y_validate = validate_df[target]

    # Apply log transformation to target variable
    y_train = np.log1p(y_train)
    y_validate = np.log1p(y_validate)

    # Scale features
    scaler_original = StandardScaler()
    X_train_original_scaled = scaler_original.fit_transform(X_train_original)
    X_validate_original_scaled = scaler_original.transform(X_validate_original)
    
    # Debug prints
    print("X_train_original_scaled shape:", X_train_original_scaled.shape)
    print("X_validate_original_scaled shape:", X_validate_original_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_validate shape:", y_validate.shape)
    
    return X_train_original_scaled, X_validate_original_scaled, y_train, y_validate

def preprocess_combined_features(train_df, validate_df, combined_features):
    """
    Preprocess combined features for training and validation data.
    Returns scaled combined features for both datasets.
    """
    # Handle missing values (simple imputation with mean)
    train_df = train_df.fillna(train_df.mean())
    validate_df = validate_df.fillna(train_df.mean())  # Use training mean for validation data
    
    # Create combined features
    train_combined_feature_data = {}
    validate_combined_feature_data = {}
    
    for feature1, feature2 in combined_features:
        feature_name = f"{feature1}_x_{feature2}"
        
        # Process training data
        train_combined_feature_data[feature_name] = train_df[feature1] * train_df[feature2]
        train_combined_feature_data[feature_name] = train_combined_feature_data[feature_name].replace(0, 1e-6)
        train_combined_feature_data[feature_name] = np.clip(train_combined_feature_data[feature_name], -1e6, 1e6)
        train_combined_feature_data[feature_name] = np.log1p(train_combined_feature_data[feature_name].abs()) * np.sign(train_combined_feature_data[feature_name])
        
        # Process validation data
        validate_combined_feature_data[feature_name] = validate_df[feature1] * validate_df[feature2]
        validate_combined_feature_data[feature_name] = validate_combined_feature_data[feature_name].replace(0, 1e-6)
        validate_combined_feature_data[feature_name] = np.clip(validate_combined_feature_data[feature_name], -1e6, 1e6)
        validate_combined_feature_data[feature_name] = np.log1p(validate_combined_feature_data[feature_name].abs()) * np.sign(validate_combined_feature_data[feature_name])
    
    # Create DataFrame for combined features
    X_train_combined = pd.DataFrame(train_combined_feature_data)
    X_validate_combined = pd.DataFrame(validate_combined_feature_data)
    
    # Scale features
    if not X_train_combined.empty:
        scaler_combined = StandardScaler()
        X_train_combined_scaled = scaler_combined.fit_transform(X_train_combined)
        X_validate_combined_scaled = scaler_combined.transform(X_validate_combined)
    else:
        X_train_combined_scaled = np.zeros((train_df.shape[0], 0))
        X_validate_combined_scaled = np.zeros((validate_df.shape[0], 0))
    
    return X_train_combined_scaled, X_validate_combined_scaled

def deal_combinations(train_data, num_combinations=8):
    """Get engineered feature combinations"""
    # Engineer features and get combinations
    train_data, all_combinations = engineer_features(train_data)
    
    # Select top combinations if we have more than requested
    if len(all_combinations) > num_combinations:
        selected_combinations = all_combinations[:num_combinations]
    else:
        # If we have fewer, repeat some important ones
        selected_combinations = all_combinations + all_combinations[:num_combinations - len(all_combinations)]
    
    return selected_combinations

def deal_combinations_static(feature_deque, num_combinations=16):
    # Static feature combinations
    return [
        ('AnnualIncome', 'Occupation_nan'),
        ('NumberofDependents', 'EducationLevel_Masters'),
        ('PreviousClaims', 'Location_Rural'),
        ('HealthScore', 'EducationLevel_Bachelors'),
        ('AnnualIncome', 'NumberofDependents'),
        ('HealthScore', 'CreditScore'),
        ('HealthScore', 'EducationLevel_Masters'),
        ('HealthScore', 'CustomerFeedback_Average'),
        ('NumberofDependents', 'Location_Rural'),
        ('AnnualIncome', 'Occupation_Unemployed'),
        ('AnnualIncome', 'InsuranceDuration'),
        ('AnnualIncome', 'CustomerFeedback_nan'),
        ('CreditScore', 'SignupYear'),
        ('MaritalStatus_nan', 'PropertyType_House'),
        ('HealthScore', 'MaritalStatus_nan'),
        ('NumberofDependents','Gender_Female')
    ]   

def run_super(s_features, test_mode=False):
    # Load combined data
    combined_data = get_data('train')
    
    # If in test mode, use a smaller subset of the data
    if test_mode:
        combined_data = combined_data.sample(frac=0.1, random_state=42)
        
    # Create custom train-validation split
    train_data, validate_data = train_test_split(combined_data, test_size=0.2, random_state=42)
    
    # Preprocess data
    X_train_original_scaled, X_validate_original_scaled, X_train_combined_scaled, X_validate_combined_scaled, y_train, y_validate = preprocess_data(
        train_data, validate_data, s_features
    )
    
    if X_train_original_scaled is None:
        print("Data preprocessing failed")
        return float('inf')
    
    num_combinations = X_train_combined_scaled.shape[1]
  
    # Create and compile model
    predictor = SuperNN(input_dim=len(s_features), num_combinations=num_combinations)
    predictor.build_model()
    predictor.compile_model(learning_rate=0.001, metrics=['mae'], loss=tf.keras.losses.Huber(delta=1.0))
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
    callbacks = [early_stopping, lr_scheduler]
    
    # Train model using the validation split from training data
    X_train_split = [X_train_original_scaled, X_train_combined_scaled]
    predictor.train_model(
        X_train_split, y_train, 
        epochs=5 if test_mode else 30, batch_size=32 if test_mode else 64,
        callbacks=callbacks, validation_split=0.15
    )
    
    # Make predictions on the separate validation dataset
    X_validate_split = [X_validate_original_scaled, X_validate_combined_scaled]
    validate_predictions = predictor.predict(X_validate_split)
    
    # Ensure predictions have the correct shape
    validate_predictions = validate_predictions.flatten()
    
    # Convert predictions back from log space
    validate_predictions = np.expm1(validate_predictions)
    y_validate_original = np.expm1(y_validate)
      
    # Calculate RMSLE score on validation data
    validate_score = safe_rmsle(y_validate_original, validate_predictions)
    print(f"Validation RMSLE score: {validate_score}")

    return validate_score

def run_ensemble(s_features, n_models=7, num_combinations=8, test_mode=False):
    """Train multiple models with improved ensemble strategy"""
    # Get separate train and validation data
    train_data = get_data('just_train', test_mode=test_mode)
    validate_data = get_data('validate', test_mode=test_mode)
    
    # First, engineer features for all datasets
    train_data_engineered, _ = engineer_features(train_data.copy())
    validate_data_engineered, _ = engineer_features(validate_data.copy())
    
    # Initialize predictions array
    predictions = np.zeros(len(validate_data))
    best_score = float('inf')
    trained_models = []
    
    # Create feature subsets for diversity in ensemble
    feature_subsets = [
        # Full feature set
        s_features,
        # Demographic features
        [f for f in s_features if f in ['Age', 'Gender_Male', 'Gender_Female', 'MaritalStatus_Married', 
                                        'MaritalStatus_Single', 'NumberofDependents']],
        # Health and insurance features
        [f for f in s_features if f in ['HealthScore', 'InsuranceDuration', 'PreviousClaims']],
        # Financial features
        [f for f in s_features if f in ['AnnualIncome', 'CreditScore']],
        # Mixed feature subset
        [f for f in s_features if f in ['Age', 'HealthScore', 'AnnualIncome', 'PreviousClaims', 'CreditScore']],
        # Additional subsets for more diversity
        [f for f in s_features if 'Education' in f or 'Occupation' in f],
        [f for f in s_features if 'Location' in f or 'Property' in f],
        [f for f in s_features if 'Customer' in f or 'Feedback' in f],
    ]
    
    # Add a default subset in case the filtering removes all features
    for i in range(len(feature_subsets)):
        if len(feature_subsets[i]) == 0:
            feature_subsets[i] = s_features[:5]  # Use first 5 features as default
    
    # Print available feature subsets
    print("Feature subsets for ensemble diversity:")
    for i, subset in enumerate(feature_subsets):
        print(f"  Subset {i+1}: {len(subset)} features")
        
    # Initialize K-Fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Define architecture rotation - avoid problematic ones by default
    architectures = [0, 1, 2, 3, 4, 5, 6]  # Skip Deep-Cross (1) by default
    
    # List to keep track of successful architectures
    successful_architectures = []
    failed_architectures = []
    
    # Track submission files and validation scores
    submission_files = []
    validation_scores = []
    
    for i in range(n_models):
        print(f"\n{'='*50}")
        print(f"TRAINING MODEL {i+1}/{n_models}")
        print(f"{'='*50}")
        
        # Rotate through architectures, avoiding those that have failed
        available_architectures = [a for a in architectures if a not in failed_architectures]
        if not available_architectures:
            print("All architectures have failed. Using default architecture 0 (RBF-SVR)")
            architecture_id = 0
        else:
            architecture_id = available_architectures[i % len(available_architectures)]
        
        # Skip problematic architecture if it previously crashed
        try:
            # Configure TensorFlow to reset memory between models
            tf.keras.backend.clear_session()
            
            # Use different feature subsets for diversity
            current_features = feature_subsets[i % len(feature_subsets)]
            print(f"Using {len(current_features)} features for this model")
            
            # Process features and train model
            combined_features = deal_combinations(train_data_engineered, num_combinations)
            
            # Create a fold for this model
            fold_idx = 0
            for train_idx, val_idx in kf.split(train_data_engineered):
                if fold_idx == i % 3:  # Use a different fold for each model
                    fold_train = train_data_engineered.iloc[train_idx]
                    fold_val = train_data_engineered.iloc[val_idx]
                    break
                fold_idx += 1
            
            X_train_original, X_validate_original, y_train, y_validate = preprocess_data(
                fold_train, validate_data_engineered, current_features
            )
            X_train_combined, X_validate_combined = preprocess_combined_features(
                fold_train, validate_data_engineered, combined_features
            )
            
            # Get actual input dimensions from data
            actual_input_dim = X_train_original.shape[1]
            actual_combo_dim = X_train_combined.shape[1]
            
            # Initialize model with ACTUAL dimensions from the data
            try:
                # Verify the actual dimensions are valid
                if actual_input_dim <= 0 or actual_combo_dim < 0:
                    raise ValueError(f"Invalid dimensions: original={actual_input_dim}, combined={actual_combo_dim}")
                
                print(f"Actual feature dimensions - Original: {actual_input_dim}, Combined: {actual_combo_dim}")
                
                predictor = SuperNN(
                    input_dim=actual_input_dim,
                    num_combinations=actual_combo_dim,
                    architecture_id=architecture_id
                )
                predictor.set_feature_combinations(combined_features, i)
                
                # Build with error catching
                try:
                    predictor.build_model()
                    
                    # Quick validation after build
                    if predictor.model is None:
                        raise ValueError("Model building returned None")
                        
                    # Verify the model's input shapes match what we expect
                    model_shapes_verified = False
                    try:
                        # Try getting shapes and verifying
                        if len(predictor.model.inputs) >= 2:
                            model_original_shape = predictor.model.inputs[0].shape[1]
                            model_combined_shape = predictor.model.inputs[1].shape[1]
                            
                            if model_original_shape != actual_input_dim or model_combined_shape != actual_combo_dim:
                                print(f"Warning: Model input shapes don't match requested dimensions")
                                print(f"Expected: Original={actual_input_dim}, Combined={actual_combo_dim}")
                                print(f"Got: Original={model_original_shape}, Combined={model_combined_shape}")
                            else:
                                model_shapes_verified = True
                    except (AttributeError, IndexError) as e:
                        print(f"Could not verify model shapes: {e}")
                        
                    if not model_shapes_verified:
                        print("Warning: Could not verify model input shapes")
                        
                except Exception as e:
                    print(f"Error building model: {e}")
                    print("Falling back to simpler architecture")
                    # Build fallback model with correct dimensions
                    predictor = SuperNN(
                        input_dim=actual_input_dim,
                        num_combinations=actual_combo_dim,
                        architecture_id=2  # Use TabNet-Residual as fallback
                    )
                    predictor.set_feature_combinations(combined_features, i)
                    predictor.build_model()
                
            except Exception as e:
                print(f"Error initializing model: {e}")
                # Last resort fallback to simplest possible model with correct dimensions
                predictor = SuperNN(
                    input_dim=actual_input_dim,
                    num_combinations=actual_combo_dim, 
                    architecture_id=4
                )
                predictor.set_feature_combinations(combined_features, i)
                predictor.build_model()
            
            # Set up optimizations for problematic architectures
            if architecture_id == 1:  # Deep-Cross
                print("Using specialized settings for Deep-Cross architecture")
                learning_rate = 0.0001  # Very low learning rate
                batch_size = 32  # Small batch size
                max_epochs = 30  # Few epochs
            else:
                learning_rates = {
                    0: 0.001,  # RBF-SVR
                    2: 0.0005, # TabNet-Residual
                    3: 0.0005, # TabNet-Attention
                    4: 0.001,  # GBM-Emulation
                    5: 0.0005, # Actuarial-Net
                    6: 0.0005  # Premium-Transformer
                }
                learning_rate = learning_rates[architecture_id]
                batch_sizes = {
                    0: 32,  # RBF-SVR
                    2: 32,  # TabNet-Residual
                    3: 32,  # TabNet-Attention
                    4: 32,  # GBM-Emulation
                    5: 32,  # Actuarial-Net
                    6: 24   # Premium-Transformer
                }
                batch_size = batch_sizes[architecture_id]
                max_epochs = 30
            
            # Compile with error catching
            try:
                predictor.compile_model(learning_rate=learning_rate)
            except Exception as e:
                print(f"Error compiling model: {e}")
                # Try compiling with default settings
                predictor.compile_model(learning_rate=0.0005)
            
            # Train the model
            print(f"\nTraining {predictor.ARCHITECTURES[architecture_id]} model {i}...")
            
            # Add a timeout to catch hanging models
            start_time = time.time()
            
            predictor.train_model(
                [X_train_original, X_train_combined],
                y_train,
                epochs=max_epochs,
                batch_size=batch_size,
                validation_split=0.2
            )
            
            # If we reach here, architecture was successful
            successful_architectures.append(architecture_id)
            
            # Make predictions
            model_predictions = predictor.predict([X_validate_original, X_validate_combined])
            predictions += model_predictions.flatten()
            
            # Add model to trained_models list - THIS WAS MISSING
            trained_models.append(predictor)
            
            print(f"\n{'*'*50}")
            print(f"MODEL {i+1}/{n_models} TRAINING COMPLETED")
            print(f"Architecture: {predictor.ARCHITECTURES[architecture_id]}")
            print(f"Validation score: {predictor.validation_score:.4f}")
            print(f"{'*'*50}")
            
            # Save if best model
            if predictor.validation_score < best_score:
                best_score = predictor.validation_score
            
            # Create and save submission immediately
            submission_path = predictor.create_submission(
                [X_validate_original, X_validate_combined],
                submission_path=f'/home/USER/Projects/ikaggle/KISS/results/submission_model_{i}.csv'
            )
            submission_files.append(submission_path)
            validation_scores.append(predictor.validation_score)
            
        except Exception as e:
            print(f"\n{'!'*50}")
            print(f"ERROR TRAINING MODEL {i+1}: {e}")
            print(f"Skipping this model and architecture...")
            print(f"{'!'*50}")
            
            # Record the failed architecture
            failed_architectures.append(architecture_id)
            
            # Continue with next model
            continue
    
    # FIXED: Change the condition to check submission files instead of trained_models
    if submission_files:  # Use submission_files to check if any models were successful
        # Average predictions - handle case when not all models trained
        if trained_models:
            predictions /= len(trained_models)
        
        # Print ensemble summary
        print(f"\n{'='*60}")
        print(f"ENSEMBLE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Number of successful models: {len(submission_files)} out of {n_models} attempted")
        if trained_models:
            architectures_used = [predictor.ARCHITECTURES[predictor.architecture_id] for predictor in trained_models]
            print(f"Architectures used: {architectures_used}")
        print(f"Successful architectures: {[SuperNN.ARCHITECTURES[a] for a in successful_architectures]}")
        print(f"Failed architectures: {[SuperNN.ARCHITECTURES[a] for a in failed_architectures]}")
        print(f"Best validation score: {best_score:.4f}")
        if trained_models:
            avg_score = np.mean([p.validation_score for p in trained_models])
            print(f"Average validation score: {avg_score:.4f}")
        print(f"{'='*60}\n")
        
        # After all models are trained, ensemble the submissions
        print("\nCreating ensemble from individual submissions...")
        weights = get_optimal_weights(submission_files, validation_scores)
        
        # Try different ensemble methods
        ensemble_methods = ['weighted', 'mean', 'median']
        ensemble_paths = []
        
        for method in ensemble_methods:
            output_path = f'/home/USER/Projects/ikaggle/KISS/results/ensemble_{method}.csv'
            ensemble_path = ensemble_submissions(
                submission_files,
                output_path=output_path,
                weights=weights,
                method=method
            )
            ensemble_paths.append(ensemble_path)
        
        # Use the weighted ensemble as the final submission
        final_path = '/home/USER/Projects/ikaggle/KISS/results/contest_submission.csv'
        import shutil
        
        # Choose which ensemble method to use for final submission based on diversity
        if diversity_score < 0.1:
            # Use diversity method for low diversity ensembles
            best_ensemble = ensemble_paths[3]  # diversity method
            print("Using diversity-based ensemble for final submission")
        else:
            # Use weighted method for normal diversity
            best_ensemble = ensemble_paths[0]  # weighted method
            print("Using weighted ensemble for final submission")
            
        shutil.copy(best_ensemble, final_path)
        print(f"\nFinal submission created at: {final_path}")
        
        return final_path
    else:
        print("No models were successfully trained.")
        # FIXED: Return emergency submission path instead of tuple
        emergency_path = '/home/USER/Projects/ikaggle/KISS/results/emergency_submission.csv'
        contest_data = get_data('contest')
        ids = range(1200000, 1200000 + len(contest_data))
        pd.DataFrame({
            'id': ids, 
            'Premium Amount': [500.0] * len(ids)
        }).to_csv(emergency_path, index=False)
        print(f"Created emergency submission at {emergency_path}")
        return emergency_path

def save_contest_predictions(predictions, output_path):
    """Save predictions with validation"""
    # Check prediction range
    pred_min = np.min(predictions)
    pred_max = np.max(predictions)
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    
    print("\nPrediction Statistics:")
    print(f"Mean: {pred_mean:.2f}")
    print(f"Std: {pred_std:.2f}")
    print(f"Min: {pred_min:.2f}")
    print(f"Max: {pred_max:.2f}")
    
    # Ensure all predictions are positive
    predictions = np.maximum(predictions, 1e-6)
    
    # Write predictions
    contest_data = get_data('contest')
    submission = pd.DataFrame({
        'Id': contest_data.index + 1200000,
        'PremiumAmount': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"\nSaved predictions to {output_path}")

def main(test_mode=False, num_combinations=8):
    """Main function for training and prediction"""
    s_features = features
    print(f"Using features: {s_features}")
    
    # Set up paths
    results_dir = '/home/USER/Projects/ikaggle/KISS/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Train new models and create submission files
    print("Training new models and creating submissions...")
    try:
        final_submission_path = run_ensemble(
            s_features, 
            n_models=7, 
            num_combinations=num_combinations, 
            test_mode=test_mode
        )
        
        print(f"\nDone! Final submission available at: {final_submission_path}")
        return final_submission_path
        
    except Exception as e:
        print(f"Error during ensemble training: {e}")
        print("Attempting to create emergency submission...")
        
        # Create an emergency submission if everything fails
        try:
            contest_data = get_data('contest')
            emergency_ids = range(1200000, 1200000 + len(contest_data))
            emergency_path = os.path.join(results_dir, 'emergency_submission.csv')
            pd.DataFrame({
                'id': emergency_ids,
                'Premium Amount': [500.0] * len(emergency_ids)
            }).to_csv(emergency_path, index=False)
            print(f"Created emergency submission at {emergency_path}")
            return emergency_path
        except Exception as e2:
            print(f"Even emergency submission creation failed: {e2}")
            return None

if __name__ == "__main__":
    import sys
    test_mode = '--test' in sys.argv
    num_combinations = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(test_mode=test_mode, num_combinations=num_combinations)


