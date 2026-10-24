import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU, Concatenate, Add, Multiply
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class DiverseModelFactory:
    """
    Factory class to create diverse models with specific architectures
    """
    @staticmethod
    def create_model(architecture_name, input_dim, seed=42):
        """Create model based on architecture name"""
        tf.random.set_seed(seed)
        
        if architecture_name == "wide_and_deep":
            return DiverseModelFactory._create_wide_and_deep(input_dim)
        elif architecture_name == "residual_net":
            return DiverseModelFactory._create_residual_net(input_dim)
        elif architecture_name == "attention_net":
            return DiverseModelFactory._create_attention_net(input_dim)
        elif architecture_name == "quantile_net":
            return DiverseModelFactory._create_quantile_net(input_dim)
        elif architecture_name == "actuarial_net":
            return DiverseModelFactory._create_actuarial_net(input_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture_name}")
    
    @staticmethod
    def _create_wide_and_deep(input_dim):
        """Create Wide and Deep model"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Wide path - linear model
        wide = Dense(1)(inputs)
        
        # Deep path - deep network
        deep = Dense(128, activation='relu')(inputs)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.3)(deep)
        deep = Dense(64, activation='relu')(deep)
        deep = BatchNormalization()(deep)
        deep = Dropout(0.3)(deep)
        deep = Dense(32, activation='relu')(deep)
        deep = Dense(1)(deep)
        
        # Combine wide and deep paths
        outputs = Add()([wide, deep])
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model, "wide_and_deep"
    
    @staticmethod
    def _create_residual_net(input_dim):
        """Create Residual Network"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Initial projection
        x = Dense(64)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # Residual blocks
        for i in range(3):
            # Store input for residual connection
            res_input = x
            
            # Process through dense layers
            x = Dense(64)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.2)(x)
            x = Dense(64)(x)
            x = BatchNormalization()(x)
            
            # Add residual connection
            x = Add()([x, res_input])
            x = LeakyReLU()(x)
        
        # Final layers
        x = Dense(32)(x)
        x = LeakyReLU()(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model, "residual_net"
    
    @staticmethod
    def _create_attention_net(input_dim):
        """Create Attention Network"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Feature attention
        attention_weights = Dense(input_dim, activation='sigmoid', name='feature_attention')(inputs)
        # FIXED: Use Keras Multiply layer instead of tf.multiply
        attended_features = Multiply()([inputs, attention_weights])
        
        # Process attended features
        x = Dense(128)(attended_features)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model, "attention_net"
    
    @staticmethod
    def _create_quantile_net(input_dim):
        """Create Quantile Regression Network"""
        # Custom quantile loss
        def quantile_loss(q, y_true, y_pred):
            e = y_true - y_pred
            return tf.reduce_mean(tf.maximum(q*e, (q-1)*e))
        
        # Create 3 quantile outputs (0.1, 0.5, 0.9)
        inputs = Input(shape=(input_dim,))
        
        # Shared layers
        x = Dense(128)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        # Separate heads for different quantiles
        q10_output = Dense(1, name='q10')(x)
        q50_output = Dense(1, name='q50')(x)
        q90_output = Dense(1, name='q90')(x)
        
        # Create model with multiple outputs
        model = Model(inputs=inputs, outputs=[q10_output, q50_output, q90_output])
        
        # Compile with different losses for each output
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=[
                lambda y_true, y_pred: quantile_loss(0.1, y_true, y_pred),
                lambda y_true, y_pred: quantile_loss(0.5, y_true, y_pred),
                lambda y_true, y_pred: quantile_loss(0.9, y_true, y_pred),
            ],
            loss_weights=[0.2, 0.6, 0.2]
        )
        
        return model, "quantile_net"
    
    @staticmethod
    def _create_actuarial_net(input_dim):
        """Create Actuarial Network with insurance-specific components"""
        inputs = Input(shape=(input_dim,))
        
        # Base risk assessment
        base_risk = Dense(64, activation='selu')(inputs)
        base_risk = BatchNormalization()(base_risk)
        
        # Risk factor pathway
        risk_factors = Dense(32, activation='selu')(inputs)
        risk_factors = BatchNormalization()(risk_factors)
        
        # Create multiplicative interactions (simulating risk tables)
        # This creates interactions between features which is common in actuarial models
        risk_interactions = Dense(32, activation='selu')(inputs)
        risk_interactions = Dense(32, activation='sigmoid')(risk_interactions)
        
        # Combine pathways
        combined = Concatenate()([base_risk, risk_factors, risk_interactions])
        
        # Final premium calculation layers
        x = Dense(64, activation='selu')(combined)
        x = BatchNormalization()(x)
        x = Dense(32, activation='selu')(x)
        x = Dense(16, activation='selu')(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='mse',
            metrics=['mae']
        )
        
        return model, "actuarial_net"

class ModelTrainer:
    """
    Class to train and evaluate diverse models
    """
    def __init__(self, results_dir='/home/USER/Projects/ikaggle/KISS/results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.models = {}
        self.history = {}
        self.predictions = {}
        self.feature_subsets = {}
        
    def train_diverse_models(self, X, y, feature_subsets, test_data, n_folds=5):
        """
        Train diverse models with different architectures and feature subsets
        
        Args:
            X: Training features DataFrame
            y: Target values
            feature_subsets: List of feature subset lists
            test_data: Test features DataFrame for predictions
            n_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with trained models and their performance metrics
        """
        # Set up architectures to train
        architectures = [
            "wide_and_deep", 
            "residual_net", 
            "attention_net", 
            "quantile_net", 
            "actuarial_net"
        ]
        
        # Log transform target for training with robust index handling
        # FIXED: Check lengths before using index to avoid mismatch errors
        try:
            if isinstance(y, np.ndarray):
                # Check if lengths match before using the index
                if len(y) == len(X):
                    y_log = pd.Series(np.log1p(y), index=X.index)
                else:
                    print(f"Warning: Length mismatch between X ({len(X)}) and y ({len(y)})")
                    # Create Series without specifying index
                    y_log = pd.Series(np.log1p(y))
            else:
                y_log = np.log1p(y)
                
            print(f"X shape: {X.shape}, y_log shape: {y_log.shape if hasattr(y_log, 'shape') else len(y_log)}")
        except Exception as e:
            print(f"Error during target transformation: {e}")
            # Emergency fallback: just use numpy array
            y_log = np.log1p(y) if isinstance(y, np.ndarray) else y.apply(np.log1p)
            print(f"Using fallback target transformation. Type: {type(y_log)}")
        
        # Set up cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store results
        results = {}
        
        # For each architecture
        for arch_name in architectures:
            print(f"\n{'='*50}")
            print(f"Training {arch_name} models")
            print(f"{'='*50}")
            
            arch_results = []
            
            # For each feature subset
            for i, features in enumerate(feature_subsets):
                subset_name = f"subset_{i}"
                
                # FIXED: Filter out features not in X or test_data
                filtered_features = [f for f in features if f in X.columns and f in test_data.columns]
                
                # Skip if too few features remain
                if len(filtered_features) < 3:
                    print(f"Skipping {subset_name} - too few valid features remain after filtering")
                    continue
                    
                print(f"\nTraining on {subset_name} with {len(filtered_features)} features")
                print(f"Filtered out {len(features) - len(filtered_features)} features not found in test data")
                
                # Extract features
                X_subset = X[filtered_features]
                
                # Store this subset
                self.feature_subsets[f"{arch_name}_{subset_name}"] = filtered_features
                
                # Set up cross-validation results
                cv_scores = []
                fold_models = []
                
                # Train on each fold
                for fold, (train_idx, val_idx) in enumerate(kf.split(X_subset)):
                    print(f"\nFold {fold+1}/{n_folds}")
                    
                    # Split data
                    # FIXED: Use numpy indexing if y_log is numpy array, otherwise use iloc
                    X_train_fold = X_subset.iloc[train_idx]
                    if isinstance(y_log, pd.Series) or isinstance(y_log, pd.DataFrame):
                        y_train_fold = y_log.iloc[train_idx]
                        y_val_fold = y_log.iloc[val_idx]
                    else:
                        y_train_fold = y_log[train_idx]
                        y_val_fold = y_log[val_idx]
                    X_val_fold = X_subset.iloc[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_fold_scaled = scaler.fit_transform(X_train_fold)
                    X_val_fold_scaled = scaler.transform(X_val_fold)
                    
                    # Create and compile model
                    model, model_name = DiverseModelFactory.create_model(
                        arch_name, 
                        input_dim=X_train_fold.shape[1], 
                        seed=42 + fold
                    )
                    
                    # Define callbacks
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
                    ]
                    
                    # Train model
                    print("Training model...")
                    history = model.fit(
                        X_train_fold_scaled,
                        y_train_fold,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_val_fold_scaled, y_val_fold),
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Evaluate on validation set
                    val_preds = model.predict(X_val_fold_scaled).flatten()
                    
                    # Convert back from log space
                    val_preds_original = np.expm1(val_preds)
                    val_true_original = np.expm1(y_val_fold)
                    
                    # Calculate RMSLE
                    from sklearn.metrics import mean_squared_log_error
                    rmsle = np.sqrt(mean_squared_log_error(val_true_original, val_preds_original))
                    cv_scores.append(rmsle)
                    
                    print(f"Fold {fold+1} RMSLE: {rmsle:.4f}")
                    
                    # Store model
                    fold_models.append({
                        'model': model,
                        'scaler': scaler,
                        'rmsle': rmsle,
                        'features': filtered_features  # FIXED: Store filtered features
                    })
                
                # Calculate average score
                avg_score = np.mean(cv_scores)
                print(f"\nAverage RMSLE across {n_folds} folds: {avg_score:.4f}")
                
                # Store best model from this subset
                best_model_idx = np.argmin(cv_scores)
                best_model = fold_models[best_model_idx]
                
                # Generate test predictions with best model
                test_subset = test_data[filtered_features]  # FIXED: Use filtered features
                test_scaled = best_model['scaler'].transform(test_subset)
                
                # For quantile models, use the median (q50) prediction
                if arch_name == "quantile_net":
                    test_preds = best_model['model'].predict(test_scaled)[1].flatten()
                else:
                    test_preds = best_model['model'].predict(test_scaled).flatten()
                    
                # Convert from log space
                test_preds_original = np.expm1(test_preds)
                
                # Store results
                model_key = f"{arch_name}_{subset_name}"
                self.models[model_key] = best_model
                self.history[model_key] = {
                    'cv_scores': cv_scores,
                    'avg_score': avg_score,
                    'best_fold': best_model_idx
                }
                self.predictions[model_key] = test_preds_original
                
                # Add to architecture results
                arch_results.append({
                    'model_key': model_key,
                    'rmsle': avg_score,
                    'n_features': len(filtered_features)  # FIXED: Count filtered features
                })
                
                # Create submission file
                self._create_submission(model_key, test_preds_original)
                
            # Store architecture results
            results[arch_name] = arch_results
            
        # Print summary of all models
        print("\n" + "="*60)
        print("MODEL TRAINING SUMMARY")
        print("="*60)
        
        # Sort models by performance
        all_models = []
        for arch_name, arch_results in results.items():
            for result in arch_results:
                all_models.append({
                    'model_key': result['model_key'],
                    'architecture': arch_name,
                    'rmsle': result['rmsle'],
                    'n_features': result['n_features']
                })
        
        # Sort by RMSLE (lower is better)
        all_models.sort(key=lambda x: x['rmsle'])
        
        # Print table
        print(f"{'Model Key':<30} {'Architecture':<15} {'RMSLE':<10} {'Features':<10}")
        print("-" * 65)
        for model in all_models:
            print(f"{model['model_key']:<30} {model['architecture']:<15} {model['rmsle']:<10.4f} {model['n_features']:<10}")
        
        # Create ensemble of best models
        best_models = all_models[:min(5, len(all_models))]
        ensemble_predictions = np.zeros(len(test_data))
        
        print("\nCreating ensemble of top 5 models:")
        for model in best_models:
            model_key = model['model_key']
            weight = 1.0 / model['rmsle']  # Weight by inverse RMSLE
            print(f"  {model_key}: weight={weight:.4f}")
            ensemble_predictions += weight * self.predictions[model_key]
        
        # Normalize predictions
        ensemble_predictions /= sum(1.0 / model['rmsle'] for model in best_models)
        
        # Save ensemble
        self._create_submission('ensemble_best5', ensemble_predictions)
        
        return results

    def _create_submission(self, model_key, predictions):
        """Create a submission file for the given model"""
        submission_path = os.path.join(self.results_dir, f"{model_key}_submission.csv")
        
        # Create dataframe with IDs from 1200000 to 1200000+n-1
        submission = pd.DataFrame({
            'id': range(1200000, 1200000 + len(predictions)),
            'Premium Amount': predictions
        })
        
        # Save to CSV
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        
        return submission_path
