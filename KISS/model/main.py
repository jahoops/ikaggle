import pandas as pd
import numpy as np
from itertools import combinations
from numpy import inf
from utils.data_manager import get_data, features, target
from model.nn import PremiumAmountPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from utils.math import safe_rmsle
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow to allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def preprocess_data(df, features, target, combined_feature=None):
    """Preprocess data: handle missing values, create combined features, scale features, and split into train/test"""
    print(f"Preprocessing data with combined feature: {combined_feature}")
    
    # Handle missing values (simple imputation with mean)
    df = df.fillna(df.mean())
    
    # Check for invalid values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Create combined feature if provided
    if combined_feature:
        feature_name = f"{combined_feature[0]}_x_{combined_feature[1]}"
        df[feature_name] = df[combined_feature[0]] * df[combined_feature[1]]
        
        # Debug: Print statistics of the combined feature before any transformation
        print(f"Combined feature '{feature_name}' statistics before transformation:")
        print(df[feature_name].describe())
        
        # Replace zeros with a small positive value
        df[feature_name] = df[feature_name].replace(0, 1e-6)
        
        # Clip the combined feature to a reasonable range
        df[feature_name] = np.clip(df[feature_name], -1e6, 1e6)
        
        # Apply log transformation to the combined feature
        df[feature_name] = np.log1p(df[feature_name].abs()) * np.sign(df[feature_name])
        
        # Debug: Print statistics of the combined feature after transformation
        print(f"Combined feature '{feature_name}' statistics after transformation:")
        print(df[feature_name].describe())
        
        # Ensure the combined feature is included in the features list
        features = features + [feature_name]
    
    # Separate features and target
    X = df[features]
    y = df[target]
    
    # Check for invalid values
    if y.isnull().any():
        print(f"Invalid data found for combined feature: {combined_feature}")
        return None, None, None, None, None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Preprocessed data with combined feature: {combined_feature}")
    return X_train, X_test, y_train, y_test, len(features)  # Return input dimension

def evaluate_feature_pair(pair, max_retries=1):
    for attempt in range(max_retries):
        try:
            print(f"Evaluating feature pair: {pair}, Attempt: {attempt + 1}")
            
            # Load training data
            train_data = get_data('just_train')
            validate_data = get_data('validate')
            
            # Preprocess data with the selected pair of features
            X_train, X_test, y_train, y_test, input_dim = preprocess_data(train_data, features, target, combined_feature=pair)
            
            # Skip invalid feature combinations
            if X_train is None or X_test is None or y_train is None or y_test is None or input_dim is None:
                print(f"Skipping invalid feature pair: {pair}")
                return (pair, None)
            
            # Debug: Print shapes of the training and test sets
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
            
            # Check for NaNs in the training data
            tf.debugging.check_numerics(X_train, message='Checking for NaNs in X_train')
            tf.debugging.check_numerics(y_train, message='Checking for NaNs in y_train')
            
            # Create and compile model
            predictor = PremiumAmountPredictor(input_dim=input_dim)
            predictor.compile_model(learning_rate=0.0001, metrics=['mae'])  # Use MAE as an additional metric
            
            # Define early stopping callback
            early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
            
            # Train model with early stopping
            predictor.train_model(X_train, y_train, epochs=10, batch_size=128, callbacks=[early_stopping])
            
            # Evaluate model
            predictor.evaluate_model(X_test, y_test)
            
            # Make predictions on validation data
            X_validate = validate_data[features].fillna(validate_data.mean())
            
            # Create combined feature for validation data
            feature_name = f"{pair[0]}_x_{pair[1]}"
            X_validate[feature_name] = X_validate[pair[0]] * X_validate[pair[1]]
            
            # Replace zeros with a small positive value
            X_validate[feature_name] = X_validate[feature_name].replace(0, 1e-6)
            
            # Clip the combined feature to a reasonable range
            X_validate[feature_name] = np.clip(X_validate[feature_name], -1e6, 1e6)
            
            # Apply log transformation to the combined feature
            X_validate[feature_name] = np.log1p(X_validate[feature_name].abs()) * np.sign(X_validate[feature_name])
            
            X_validate_scaled = StandardScaler().fit_transform(X_validate)
            validate_predictions = predictor.predict(X_validate_scaled)
            
            # Ensure predictions have the correct shape
            validate_predictions = validate_predictions.flatten()
            
            # Debug: Print statistics of the predictions
            print(f"Predictions statistics for {pair}:")
            print(pd.Series(validate_predictions).describe())
            
            # Check for invalid predictions
            if np.any(np.isnan(validate_predictions)) or np.any(np.isinf(validate_predictions)):
                print(f"Invalid predictions found for {pair}")
                continue
            
            # Calculate RMSLE score on validation data
            validate_score = safe_rmsle(validate_data[target], validate_predictions)

            if validate_score == inf:
                print(f"Invalid RMSLE score found for {pair}")
            else:
                # Log the results to the file
                with open('/home/USER/Projects/ikaggle/KISS/data/feature_combination_results.txt', 'a') as log_file:
                    log_file.write(f"{pair}, {validate_score}\n")

            return (pair, validate_score)
        except tf.errors.InvalidArgumentError as e:
            print(f"Error evaluating feature pair {pair} on attempt {attempt + 1}: {e}")
            continue
        except Exception as e:
            print(f"Error evaluating feature pair {pair} on attempt {attempt + 1}: {e}")
    return (pair, None)

def main():
    # Generate all possible pairs of features
    feature_pairs = list(combinations(features, 2))
    # Read feature_combination_results.txt file
    
    # while feature_pairs is not empty
    while len(feature_pairs) > 0:
        pairs_to_remove = []
        with open('/home/USER/Projects/ikaggle/KISS/data/feature_combination_results.txt', 'r') as log_file:
            lines = log_file.readlines()
            for line in lines[1:]:  # Skip the header line
                if line.strip():  # Skip blank lines
                    parts = line.strip().rsplit(',', 1)
                    if len(parts) == 2:
                        pair, score = parts
                        pairs_to_remove.append(eval(pair))
        
        for pair in pairs_to_remove:
            if pair in feature_pairs:
                feature_pairs.remove(pair)
                
        results = []
        for pair in feature_pairs:
            result = evaluate_feature_pair(pair)
            if result[1] is not None:
                results.append(result)
                print(result)
    
    # Sort results by RMSLE score
    results.sort(key=lambda x: x[1])
    
    # Print the best feature pairs
    print("Top 10 feature pairs with the lowest RMSLE scores:")
    for pair, score in results[:10]:
        print(f"Feature pair: {pair}, RMSLE score: {score}")

if __name__ == "__main__":
    main()