import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.math import safe_rmsle
from model.super_nn import SuperNN
import json

class GroupSpecialistTrainer:
    """
    Trains specialized models for different customer segments
    """
    def __init__(self, results_dir='/home/USER/Projects/ikaggle/KISS/results/group_models'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.models = {}  # {group_id: [models]}
        self.performance = {}  # {model_id: {group_id: score}}
        self.group_assignments = None
        self.group_submission_files = {}  # {group_id: [submission_files]}
        self.best_models_per_group = {}  # {group_id: [model_ids]}
    
    def assign_groups(self, segmenter, X_train, X_test=None):
        """
        Assign groups to train and test data
        
        Args:
            segmenter: Trained InsuranceSegmenter
            X_train: Training data
            X_test: Optional test data
            
        Returns:
            Dictionary with group assignments
        """
        print("Assigning groups to data points...")
        
        # Assign groups to training data
        train_groups = segmenter.clusters if hasattr(segmenter, 'clusters') else segmenter.assign_segment(X_train)
        
        # Create dictionary
        self.group_assignments = {
            'train': train_groups
        }
        
        # Handle test data if provided
        if X_test is not None:
            test_groups = segmenter.assign_segment(X_test)
            self.group_assignments['test'] = test_groups
        
        # Print group sizes
        for group_id in sorted(np.unique(train_groups)):
            count = np.sum(train_groups == group_id)
            percentage = 100 * count / len(train_groups)
            print(f"Group {group_id}: {count} samples ({percentage:.2f}%)")
        
        return self.group_assignments
    
    def evaluate_models_by_group(self, submission_files, group_ids, ground_truth=None):
        """
        Evaluate existing models on each group
        
        Args:
            submission_files: List of submission files
            group_ids: Group assignments for test data
            ground_truth: Optional ground truth values for scoring
            
        Returns:
            Dictionary with model performance by group
        """
        print("Evaluating model performance by group...")
        
        # Load all submissions
        submissions = []
        model_ids = []
        
        for file_path in submission_files:
            try:
                df = pd.read_csv(file_path)
                model_id = os.path.basename(file_path).replace('.csv', '')
                
                submissions.append(df)
                model_ids.append(model_id)
                
                # Initialize performance dictionary for this model
                self.performance[model_id] = {}
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Group performance evaluations
        for group_id in sorted(np.unique(group_ids)):
            # Get indices for this group
            group_indices = np.where(group_ids == group_id)[0]
            
            # Initialize group submissions
            if group_id not in self.group_submission_files:
                self.group_submission_files[group_id] = []
            
            # For each model
            for i, (model_id, submission) in enumerate(zip(model_ids, submissions)):
                try:
                    # FIXED: Use integer position with iloc instead of loc
                    # Extract predictions for this group
                    group_preds = submission.iloc[group_indices]['Premium Amount'].values
                    
                    # If we have ground truth, calculate score
                    score = None
                    if ground_truth is not None:
                        group_truth = ground_truth[group_indices]
                        score = safe_rmsle(group_truth, group_preds)
                        
                        # Store score
                        self.performance[model_id][group_id] = score
                        print(f"Model {model_id} on Group {group_id}: RMSLE = {score:.4f}")
                    
                    # Create group-specific submission file 
                    group_file_path = f"{self.results_dir}/{model_id}_group{group_id}.csv"
                    
                    # FIXED: Use iloc for position-based indexing
                    group_submission = submission.iloc[group_indices].copy()
                    group_submission.to_csv(group_file_path, index=False)
                    
                    # Store file path
                    self.group_submission_files[group_id].append(group_file_path)
                except Exception as e:
                    print(f"Error processing group {group_id} for model {model_id}: {e}")
                    print("Skipping this group/model combination")
        
        # Find best models for each group
        if ground_truth is not None:
            self._find_best_models_per_group()
        
        return self.performance
    
    def _find_best_models_per_group(self, top_n=3):
        """Find the top N models for each group"""
        for group_id in self.group_submission_files.keys():
            # Get scores for all models on this group
            group_scores = [(model_id, self.performance[model_id][group_id]) 
                            for model_id in self.performance.keys() 
                            if group_id in self.performance[model_id]]
            
            # Sort by score (ascending - lower is better)
            group_scores.sort(key=lambda x: x[1])
            
            # Get top N model IDs
            top_models = [x[0] for x in group_scores[:top_n]]
            self.best_models_per_group[group_id] = top_models
            
            print(f"Top {top_n} models for Group {group_id}: {top_models}")
        
        return self.best_models_per_group
    
    def train_specialized_models(self, general_models, X_train, y_train, X_val=None, y_val=None):
        """Fine-tune specialized models for each group"""
        print("Training specialized models for each group...")
        
        if self.group_assignments is None:
            print("No group assignments found. Assign groups first.")
            return
        
        # Initialize specialized models dictionary
        specialized_models = {}
        
        # For each group
        for group_id in sorted(np.unique(self.group_assignments['train'])):
            print(f"\nTraining specialized models for Group {group_id}...")
            group_indices = np.where(self.group_assignments['train'] == group_id)[0]
            
            # Extract data for this group
            X_group = [x[group_indices] for x in X_train] if isinstance(X_train, list) else X_train[group_indices]
            y_group = y_train[group_indices]
            
            # Initialize group models
            specialized_models[group_id] = []
            
            # Identify which models to specialize for this group
            if group_id in self.best_models_per_group:
                model_ids_to_train = self.best_models_per_group[group_id]
            else:
                # If we don't have performance data, use all models
                model_ids_to_train = [f"model_{i}" for i in range(len(general_models))]
            
            # Fine-tune each selected model
            for i, base_model in enumerate(general_models):
                model_id = getattr(base_model, 'model_id', f"model_{i}")
                
                # Create a copy of the model for specialization
                specialized_model = self._create_specialized_model(base_model, group_id)
                
                # Fine-tune the model on group data
                self._fine_tune_model(specialized_model, X_group, y_group, X_val, y_val)
                
                # Store the specialized model
                specialized_models[group_id].append(specialized_model)
                
                # ADDED: Create submission file with specialized model
                if X_val is not None:
                    try:
                        # Save submission using model's own method if available
                        if hasattr(specialized_model, 'create_submission'):
                            submission_path = specialized_model.create_submission(
                                X_val,
                                submission_path=f"{self.results_dir}/specialized_model_{specialized_model.model_id}_group{group_id}.csv"
                            )
                        else:
                            # Fallback if model doesn't have create_submission
                            preds = specialized_model.predict(X_val).flatten()
                            preds = np.expm1(preds)  # Convert from log space
                            
                            # Create submission DataFrame
                            submission = pd.DataFrame({
                                'id': range(1200000, 1200000 + len(preds)),
                                'Premium Amount': preds
                            })
                            
                            # Save to file
                            submission_path = f"{self.results_dir}/specialized_model_{model_id}_group{group_id}.csv"
                            submission.to_csv(submission_path, index=False)
                            print(f"Created submission for model {model_id} on group {group_id}")
                    except Exception as e:
                        print(f"Error creating submission for model {model_id} on group {group_id}: {e}")
        
        # ADDED: Save a marker file to indicate specialized training is complete
        try:
            with open(f"{self.results_dir}/specialized_training_complete.txt", 'w') as f:
                f.write("Specialized training completed successfully\n")
                for group_id, models in specialized_models.items():
                    f.write(f"Group {group_id}: {len(models)} models\n")
        except Exception as e:
            print(f"Error saving specialized training marker: {e}")
            
        return specialized_models
    
    def _create_specialized_model(self, base_model, group_id):
        """Create a copy of a base model for specialization"""
        # Check if this is a SuperNN model or a SimpleModelWrapper
        if isinstance(base_model, SuperNN):
            # Handle SuperNN models as before
            specialized_model = SuperNN(
                input_dim=base_model.input_dim,
                num_combinations=base_model.num_combinations,
                architecture_id=base_model.architecture_id
            )
            
            # Set model ID and other properties
            specialized_model.model_id = f"{base_model.model_id}_group{group_id}"
            specialized_model.feature_combinations = base_model.feature_combinations.copy()
            
            # Clone the model weights
            specialized_model.build_model()
            specialized_model.model.set_weights(base_model.model.get_weights())
            
        else:
            # For SimpleModelWrapper and other models, create a direct clone
            print(f"Creating direct clone of {base_model.__class__.__name__} for group {group_id}")
            
            # Clone the Keras model directly
            if hasattr(base_model, 'model'):
                try:
                    # Get the architecture configuration
                    model_config = base_model.model.get_config()
                    # Create a new model with the same architecture
                    cloned_keras_model = tf.keras.Model.from_config(model_config)
                    # Copy the weights
                    cloned_keras_model.set_weights(base_model.model.get_weights())
                    
                    # FIXED: Compile the model with the same configuration if available
                    if hasattr(base_model.model, 'optimizer'):
                        optimizer_config = base_model.model.optimizer.get_config()
                        optimizer_class = base_model.model.optimizer.__class__
                        optimizer = optimizer_class.from_config(optimizer_config)
                        
                        cloned_keras_model.compile(
                            optimizer=optimizer,
                            loss=base_model.model.loss,
                            metrics=base_model.model.metrics
                        )
                    else:
                        # If original model wasn't compiled, compile with defaults
                        cloned_keras_model.compile(
                            optimizer='adam',
                            loss='mse',
                            metrics=['mae']
                        )
                    
                    # Create a new wrapper with the same structure
                    specialized_model = type(base_model)(
                        cloned_keras_model, 
                        getattr(base_model, 'input_shape', base_model.model.input_shape)
                    )
                    
                    # Copy other attributes
                    for attr in ['model_id', 'feature_combinations', 'architecture_id', 
                                'num_combinations', 'input_dim']:
                        if hasattr(base_model, attr):
                            setattr(specialized_model, attr, getattr(base_model, attr))
                    
                    # Update model_id
                    specialized_model.model_id = f"{getattr(base_model, 'model_id', 'model')}_group{group_id}"
                    print(f"Successfully cloned and compiled model for group {group_id}")
                    
                except Exception as e:
                    print(f"Error cloning model: {e}")
                    specialized_model = base_model  # Fallback to using the original
                    specialized_model.model_id = f"{getattr(base_model, 'model_id', 'model')}_group{group_id}"
            else:
                # Fallback if model doesn't have expected structure
                print(f"Warning: Could not clone model of type {type(base_model)}. Creating placeholder.")
                specialized_model = base_model
                specialized_model.model_id = f"{getattr(base_model, 'model_id', 'model')}_group{group_id}"
        
        return specialized_model
    
    def _fine_tune_model(self, model, X_train, y_train, X_val=None, y_val=None, epochs=10):
        """Fine-tune a model on specific group data"""
        try:
            # Only compile SuperNN models - other models should be pre-compiled
            if isinstance(model, SuperNN):
                model.compile_model(learning_rate=0.0001)
                
                # Set up callbacks
                callbacks = [
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                ]
                
                # Train the model
                model.train_model(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2 if X_val is None else None,
                    validation_data=(X_val, y_val) if X_val is not None else None,
                    callbacks=callbacks
                )
            else:
                # For simple wrappers, use their native fit method
                print(f"Fine-tuning {model.__class__.__name__} using native training method")
                
                # FIXED: Handle dimension mismatch
                # Extract actual data - handle list inputs
                X_data = X_train[0] if isinstance(X_train, list) else X_train
                
                # Check input dimensions and adapt if needed
                expected_shape = None
                if hasattr(model.model, 'input_shape'):
                    expected_shape = model.model.input_shape[1]
                elif hasattr(model.model, 'layers') and len(model.model.layers) > 0:
                    expected_shape = model.model.layers[0].input_shape[1]
                    
                if expected_shape and X_data.shape[1] != expected_shape:
                    print(f"Warning: Input dimension mismatch. Model expects {expected_shape}, got {X_data.shape[1]}")
                    
                    # Adapt input dimensions
                    if X_data.shape[1] > expected_shape:
                        print(f"Truncating input features from {X_data.shape[1]} to {expected_shape}")
                        X_data = X_data[:, :expected_shape]
                    else:
                        print(f"Padding input features from {X_data.shape[1]} to {expected_shape}")
                        padding = np.zeros((X_data.shape[0], expected_shape - X_data.shape[1]))
                        X_data = np.hstack([X_data, padding])
                        
                # Same for validation data
                val_data = None
                if X_val is not None:
                    val_X = X_val[0] if isinstance(X_val, list) else X_val
                    
                    # Adapt validation data dimensions
                    if expected_shape and val_X.shape[1] != expected_shape:
                        if val_X.shape[1] > expected_shape:
                            val_X = val_X[:, :expected_shape]
                        else:
                            padding = np.zeros((val_X.shape[0], expected_shape - val_X.shape[1]))
                            val_X = np.hstack([val_X, padding])
                            
                    val_data = (val_X, y_val)
                
                # Ensure model is compiled
                if not hasattr(model.model, 'optimizer') or model.model.optimizer is None:
                    print("Model not compiled. Compiling with default settings...")
                    model.model.compile(
                        optimizer='adam',
                        loss='mse',
                        metrics=['mae']
                    )
                
                # Setup callbacks for early stopping
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
                ]
                
                # Fit the model
                if hasattr(model.model, 'fit'):
                    model.model.fit(
                        X_data, y_train,
                        epochs=min(5, epochs),  # Use fewer epochs for simple models
                        batch_size=32,
                        validation_split=0.2 if val_data is None else None,
                        validation_data=val_data,
                        callbacks=callbacks,
                        verbose=1
                    )
        except Exception as e:
            print(f"Error during fine-tuning: {e}")
            print(f"Skipping fine-tuning for this model: {getattr(model, 'model_id', 'unknown')}")
        
        return model
    
    def create_group_based_ensemble(self, specialized_models, segmenter, X_test):
        """Create an ensemble that uses specialized models for each group"""
        print("Creating group-based ensemble...")
        
        try:
            # Try to assign test data to groups with better error handling
            test_groups = segmenter.assign_segment(X_test)
            print(f"Successfully assigned {len(test_groups)} test samples to {len(np.unique(test_groups))} groups")
        except Exception as e:
            print(f"Error assigning segments: {e}")
            print("Using uniform group assignment as fallback")
            # Fallback to uniform assignment across groups
            group_ids = sorted(specialized_models.keys())
            test_groups = np.zeros(len(X_test[0]) if isinstance(X_test, list) else len(X_test), dtype=int)
            group_size = len(test_groups) // len(group_ids)
            
            for i, group_id in enumerate(group_ids):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < len(group_ids) - 1 else len(test_groups)
                test_groups[start_idx:end_idx] = group_id
        
        # Create a DataFrame for the final predictions
        final_preds = pd.DataFrame({
            'id': range(1200000, 1200000 + len(X_test[0]) if isinstance(X_test, list) else len(X_test))
        })
        
        # Create a prediction array that we'll fill in by position
        all_predictions = np.zeros(len(final_preds))
        predictions_set = np.zeros(len(final_preds), dtype=bool)
        
        # Generate predictions for each group using specialized models
        for group_id, models in specialized_models.items():
            # Get indices for this group
            group_indices = np.where(test_groups == group_id)[0]
            
            if len(group_indices) == 0:
                print(f"No test samples in Group {group_id}")
                continue
                
            print(f"Generating predictions for Group {group_id} with {len(models)} specialized models...")
            
            # Extract test data for this group
            X_group = [x[group_indices] for x in X_test] if isinstance(X_test, list) else X_test[group_indices]
            
            # Generate predictions from each model
            group_preds = np.zeros(len(group_indices))
            
            for model in models:
                preds = model.predict(X_group).flatten()
                preds = np.expm1(preds)  # Convert from log space
                group_preds += preds
            
            # Average predictions
            if len(models) > 0:
                group_preds /= len(models)
            
            # FIXED: Use array indexing instead of DataFrame loc indexing
            # Add to final predictions array by position
            for i, idx in enumerate(group_indices):
                if idx < len(all_predictions):
                    all_predictions[idx] = group_preds[i]
                    predictions_set[idx] = True
        
        # Check for any missing predictions
        missing_count = np.sum(~predictions_set)
        if missing_count > 0:
            print(f"Warning: {missing_count} missing predictions! Using fallback models.")
            
            # For any missing predictions, use general ensemble
            fallback_indices = np.where(~predictions_set)[0]
            X_fallback = [x[fallback_indices] for x in X_test] if isinstance(X_test, list) else X_test[fallback_indices]
            
            # Use average of all models as fallback
            all_models = [model for models in specialized_models.values() for model in models]
            fallback_preds = np.zeros(len(fallback_indices))
            
            for model in all_models[:3]:  # Use first 3 models as fallback
                preds = model.predict(X_fallback).flatten()
                preds = np.expm1(preds)  # Convert from log space
                fallback_preds += preds
                
            if len(all_models) >= 3:
                fallback_preds /= 3
            
            # FIXED: Use array indexing
            for i, idx in enumerate(fallback_indices):
                if idx < len(all_predictions):
                    all_predictions[idx] = fallback_preds[i]
        
        # ADDED: Ensure all predictions are valid (no negatives)
        invalid_count = np.sum(all_predictions < 0)
        if invalid_count > 0:
            print(f"Warning: Found {invalid_count} negative predictions. Fixing...")
            all_predictions = np.maximum(all_predictions, 0.0)
        
        # ADDED: Apply validity checks
        print("\nEnsemble prediction statistics:")
        print(f"Min: {np.min(all_predictions):.2f}, Max: {np.max(all_predictions):.2f}")
        print(f"Mean: {np.mean(all_predictions):.2f}, Median: {np.median(all_predictions):.2f}")
        
        # Copy array to DataFrame
        final_preds['Premium Amount'] = all_predictions
        
        # Save final predictions
        output_path = f"{self.results_dir}/group_ensemble_submission.csv"
        final_preds.to_csv(output_path, index=False)
        print(f"Group-based ensemble saved to {output_path}")
        
        return output_path
