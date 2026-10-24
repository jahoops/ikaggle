import tensorflow as tf
from ikaggle.AAA.experiments.feature_engineering import encode_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from ikaggle.AAA.utils.validation import validate_submission
from ikaggle.AAA.utils.feature_inspector import inspect_feature_correlations, check_data_leakage, inspect_derived_features
from ikaggle.AAA.utils.feature_inspector import inspect_numerical_relationships, check_model_validation
from ikaggle.AAA.config.paths import DATA_PATHS, CACHE_PATHS
import time
from tensorflow.keras.mixed_precision import LossScaleOptimizer

# Update GPU strategy and mixed precision setup
gpu_strategy = tf.distribute.OneDeviceStrategy("/GPU:0")

def configure_model_gpu():
    """Configure GPU and mixed precision"""
    # Set XLA and mixed precision options
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.keras.backend.set_floatx('float16')
    
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Enable XLA optimization
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': True,
        'constant_folding': True,
        'shape_optimization': True,
        'remapping': True,
        'arithmetic_optimization': True,
        'dependency_optimization': True,
        'loop_optimization': True,
        'function_optimization': True,
        'debug_stripper': True,
    })

# Call configuration
configure_model_gpu()

def create_dataset(X, y, batch_size=64, shuffle=True):
    """Create dataset with proper mixed precision types"""
    X = tf.convert_to_tensor(X, dtype='float16')
    y = tf.convert_to_tensor(y, dtype='float32')
    
    # Create dataset directly from tensors
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Configure options first
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    
    # Apply options
    dataset = dataset.with_options(options)
    
    if shuffle:
        # Use deterministic shuffle
        dataset = dataset.shuffle(
            buffer_size=min(len(X), 10000),
            seed=42,
            reshuffle_each_iteration=True
        )
    
    # Batch and prefetch
    dataset = dataset.batch(
        batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def normalize_features(X):
    """Normalize features to 0-1 range with robust scaling"""
    # Ensure we're working with a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    X_norm = pd.DataFrame(index=X.index)
    
    # Handle each column separately
    for col in X.columns:
        values = X[col].values
        
        # Skip if no variation
        if np.std(values) < 1e-8:
            X_norm[col] = values
            continue
            
        # Check if column is binary (0-1)
        unique_vals = set(values)
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            X_norm[col] = values
            continue
            
        # Robust scaling for continuous features
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            iqr = np.std(values)
        if iqr == 0:
            iqr = 1.0
            
        # Center around median and scale by IQR
        median = np.median(values)
        normalized = (values - median) / (2 * iqr)
        
        # Shift to 0-1 range
        X_norm[col] = 0.5 + normalized
        X_norm[col] = np.clip(X_norm[col], 0, 1)
        
    return X_norm

def test_model_predictions(model, X_train, verbose=False):
    """Run a single set of prediction tests"""
    n_features = X_train.shape[1]
    
    # Test cases
    zeros = np.zeros((5, n_features), dtype=np.float32)
    ones = np.ones((5, n_features), dtype=np.float32)
    random = np.random.rand(5, n_features).astype(np.float32)
    
    print("\nModel Response Tests:")
    print("\n1. Zero inputs:", model.predict(zeros, batch_size=5).ravel())
    print("\n2. One inputs:", model.predict(ones, batch_size=5).ravel())
    print("\n3. Random inputs:", model.predict(random, batch_size=5).ravel())
    print("\n4. Actual sample:", model.predict(X_train[:5], batch_size=5).ravel())

def fitness_function(individual, X, y, verbose=False, return_predictions=False):
    """Evaluate individual fitness using parallel processing"""
    try:
        if not return_predictions and hasattr(individual, 'fitness') and individual.fitness.valid:
            return individual.fitness.values
        
        # Handle input data - simplified
        X_data = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if 'id' in X_data.columns:
            X_data = X_data.drop(columns=['id'])
            
        if verbose:
            print(f"X shape before processing: {X_data.shape}")
            print(f"Columns: {X_data.columns.tolist()}")
        
        # Get features with explicit error handling
        try:
            features = encode_features(individual, X_data)
            if features is None:
                print("ERROR: Feature encoding returned None")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
        except Exception as e:
            print(f"ERROR: Feature encoding failed: {str(e)}")
            return (float('inf'),) if not return_predictions else (float('inf'), None)
            
        # Convert to DataFrame with validation
        try:
            features_df = pd.DataFrame(features, index=X_data.index)
            if features_df.empty:
                print("ERROR: Empty features DataFrame")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
            if features_df.isnull().any().any():
                print("ERROR: Features contain null values")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
        except Exception as e:
            print(f"ERROR: Feature DataFrame creation failed: {str(e)}")
            return (float('inf'),) if not return_predictions else (float('inf'), None)
            
        # Normalize features with validation
        try:
            features_df = normalize_features(features_df)
            if features_df.isnull().any().any():
                print("ERROR: Normalization produced null values")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
        except Exception as e:
            print(f"ERROR: Feature normalization failed: {str(e)}")
            return (float('inf'),) if not return_predictions else (float('inf'), None)
            
        # Convert to numpy arrays with validation
        try:
            X_arr = features_df.values.astype(np.float32)
            if np.isnan(X_arr).any() or np.isinf(X_arr).any():
                print("ERROR: Invalid values in feature array")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
        except Exception as e:
            print(f"ERROR: Array conversion failed: {str(e)}")
            return (float('inf'),) if not return_predictions else (float('inf'), None)
            
        # Process target data - Fixed version
        try:
            # Convert to numpy array first
            y_array = np.array(y.values if isinstance(y, pd.Series) else y)
            y_log = np.log1p(y_array)
            y_min, y_max = y_log.min(), y_log.max()
            y_norm = (y_log - y_min) / (y_max - y_min)
            y_norm = y_norm.reshape(-1, 1)  # Now reshape will work
            
            if np.isnan(y_norm).any() or np.isinf(y_norm).any():
                print("ERROR: Invalid values in target array")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
                
        except Exception as e:
            print(f"ERROR: Target processing failed: {str(e)}")
            print(f"Target type: {type(y)}, shape: {getattr(y, 'shape', 'no shape')}")
            return (float('inf'),) if not return_predictions else (float('inf'), None)
            
        # Convert to tensors and create train/val split
        with tf.device('/GPU:0'):
            try:
                # Convert to numpy arrays first
                X_arr = features_df.values.astype(np.float32)
                y_arr = y_norm.astype(np.float32)
                
                # Create train/val split
                val_size = int(0.2 * len(X_arr))
                indices = np.random.permutation(len(X_arr))
                train_idx, val_idx = indices[val_size:], indices[:val_size]
                
                # Split the data
                X_train = X_arr[train_idx]
                y_train = y_arr[train_idx]
                X_val = X_arr[val_idx]
                y_val = y_arr[val_idx]
                
                # Create datasets
                train_dataset = create_dataset(X_train, y_train, batch_size=2048)
                val_dataset = create_dataset(X_val, y_val, batch_size=2048, shuffle=False)
                
                # Build and compile model with validation metrics
                model = build_model(input_shape=(X_arr.shape[1],), feature_cols=None)
                model.compile(
                    optimizer=tf.keras.optimizers.AdamW(
                        learning_rate=1e-3,
                        weight_decay=1e-5,
                        clipnorm=0.5
                    ),
                    loss=tf.keras.losses.Huber(delta=1.0),
                )
                
                # Create custom callback to monitor both training and validation
                class MonitorCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        # Add validation loss explicitly
                        if self.model.history.history.get('val_loss') is not None:
                            logs['val_loss'] = self.model.history.history['val_loss'][-1]
                        if self.model.history.history.get('val_mae') is not None:
                            logs['val_mae'] = self.model.history.history['val_mae'][-1]
                
                # Train with custom callback and validation
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=100,
                    verbose=0,
                    callbacks=[
                        MonitorCallback(),
                        tf.keras.callbacks.EarlyStopping(
                            monitor='loss',           # Changed to monitor training loss
                            mode='min',
                            patience=5,
                            restore_best_weights=True,
                            verbose=1 if verbose else 0
                        ),
                        tf.keras.callbacks.ReduceLROnPlateau(
                            monitor='loss',           # Changed to monitor training loss
                            mode='min',
                            factor=0.2,
                            patience=3,
                            min_lr=1e-6,
                            verbose=1 if verbose else 0
                        )
                    ]
                )
                
                # Print training summary if verbose
                if verbose:
                    print("\nTraining history:")
                    for metric in history.history:
                        print(f"{metric}: {history.history[metric][-1]:.4f}")
                
                # Get predictions
                preds = get_predictions(model, X_arr)
                
                if np.isnan(preds).any() or np.isinf(preds).any():
                    print("ERROR: Invalid predictions")
                    return (float('inf'),) if not return_predictions else (float('inf'), None)
                # print preds in GREEN
                print("\033[92mPredictions:")
                print(preds)  # Print the predictions for better visibility
                # Transform predictions
                preds = np.expm1(preds * (y_max - y_min) + y_min)
                
                if return_predictions:
                    return float('inf'), preds.ravel()
                else:
                    # Use validation loss for fitness if available, otherwise use training loss
                    val_loss = (history.history['val_loss'][-1] 
                              if 'val_loss' in history.history 
                              else history.history['loss'][-1])
                    return (float(val_loss),)
                    
            except Exception as e:
                print(f"ERROR: Model training failed: {str(e)}")
                return (float('inf'),) if not return_predictions else (float('inf'), None)
                
    except Exception as e:
        print(f"ERROR in fitness function: {str(e)}")
        import traceback
        traceback.print_exc()
        return (float('inf'),) if not return_predictions else (float('inf'), None)

def calculate_rmsle(y_true, y_pred):
    """Calculate RMSLE with improved numerical stability"""
    # Convert and clean inputs
    y_true = np.maximum(np.array(y_true).ravel(), 1.0)
    y_pred = np.maximum(np.array(y_pred).ravel(), 1.0)
    
    # Log transform with offset for stability
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    
    # Calculate error
    squared_errors = np.square(log_true - log_pred)
    rmsle = float(np.sqrt(np.mean(squared_errors)))
    
    # Add ratio analysis
    ratios = y_pred / y_true
    print("\nPrediction/Target Ratios:")
    print(f"  Mean ratio: {np.mean(ratios):.4f}")
    print(f"  Median ratio: {np.median(ratios):.4f}")
    print(f"  Ratio std: {np.std(ratios):.4f}")
    # print rmsle in GREEN
    print("\033[92mRMSLE: ", rmsle)
    # print first 5 predictions
    print("\nFirst 5 Predictions:")
    print(y_pred[:5])
    # print first 5 targets
    print("\nFirst 5 Targets:")
    print(y_true[:5], "\033[0m")
    return rmsle

class TrainableModel(tf.keras.Model):
    """Custom model with reduced retracing and proper mixed precision"""
    def __init__(self, input_shape, name='evolution_model'):
        super().__init__(name=name)
        
        # Define input shape explicitly
        self.input_shape_spec = input_shape
        
        # Define layers as class attributes
        self.dense1 = tf.keras.layers.Dense(
            256, kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42),
            dtype='float16'
        )
        self.bn1 = tf.keras.layers.BatchNormalization(dtype='float16')
        self.prelu1 = tf.keras.layers.PReLU(dtype='float16')
        self.drop1 = tf.keras.layers.Dropout(0.3, seed=42, dtype='float16')
        
        self.dense2 = tf.keras.layers.Dense(
            128,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
            dtype='float16'
        )
        self.bn2 = tf.keras.layers.BatchNormalization(dtype='float16')
        self.prelu2 = tf.keras.layers.PReLU(dtype='float16')
        self.drop2 = tf.keras.layers.Dropout(0.2, seed=44, dtype='float16')
        
        self.dense3 = tf.keras.layers.Dense(
            64,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=45),
            dtype='float16'
        )
        self.bn3 = tf.keras.layers.BatchNormalization(dtype='float16')
        self.prelu3 = tf.keras.layers.PReLU(dtype='float16')
        self.drop3 = tf.keras.layers.Dropout(0.1, seed=46, dtype='float16')
        
        self.output_dense = tf.keras.layers.Dense(
            1,
            activation='softplus',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=47),
            dtype='float32'
        )
    
    # Add input shape specification
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape_spec': self.input_shape_spec
        })
        return config

    def call(self, inputs, training=False):
        """Forward pass with explicit mixed precision handling"""
        try:
            x = tf.cast(inputs, dtype=tf.float16)  # Cast inputs to float16
            
            x = self.dense1(x)
            x = self.bn1(x, training=training)
            x = self.prelu1(x)
            x = self.drop1(x, training=training)
            
            x = self.dense2(x)
            x = self.bn2(x, training=training)
            x = self.prelu2(x)
            x = self.drop2(x, training=training)
            
            x = self.dense3(x)
            x = self.bn3(x, training=training)
            x = self.prelu3(x)
            x = self.drop3(x, training=training)
            
            # Output layer remains float32
            output = self.output_dense(x)
            return output
        except Exception as e:
            print(f"Error in TrainableModel.call: {e}")
            raise  # Re-raise the exception after logging

def build_model(input_shape, feature_cols=None, verbose=False):
    """Build model with proper mixed precision configuration"""
    with gpu_strategy.scope():
        # Create model
        model = TrainableModel(input_shape)
        
        # Create optimizer with loss scaling
        base_optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-3,
            weight_decay=1e-5,
            clipnorm=0.5
        )
        optimizer = LossScaleOptimizer(base_optimizer)
        
        # Create metrics inside strategy scope
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.MeanSquaredError()
        ]
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=metrics,
            jit_compile=True
        )
        
        # Build model
        dummy_input = tf.zeros((1,) + input_shape, dtype='float16')
        _ = model(dummy_input)
        
        if verbose:
            model.summary()
            
        return model

def compile_model(model, learning_rate=5e-4):
    """Compile model with optimized configuration"""
    with gpu_strategy.scope():
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-4,
            clipnorm=1.0,
            epsilon=1e-7,
            jit_compile=True
        )
        
        # Use static shapes for loss and metrics
        loss = tf.keras.losses.Huber(
            delta=1.0, 
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
        
        metrics = [
            tf.keras.metrics.MeanAbsoluteError()
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=True,
            # Experimental options to reduce retracing
            experimental_steps_per_execution=10
        )
        return model

def get_predictions(model, X, batch_size=4096):
    """Get predictions with consistent tensor shapes"""
    # Convert to numpy array with consistent dtype
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = X.astype(np.float32)
    
    # Pre-allocate output array
    n_samples = len(X)
    predictions = np.zeros(n_samples, dtype=np.float32)
    
    # Process full batches first
    n_full_batches = n_samples // batch_size
    for i in range(n_full_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = tf.convert_to_tensor(X[start_idx:end_idx], dtype=tf.float32)
        predictions[start_idx:end_idx] = model_predict_fn(model, batch).numpy().ravel()
    
    # Handle final partial batch if it exists
    if n_samples % batch_size != 0:
        start_idx = n_full_batches * batch_size
        final_batch = X[start_idx:]
        # Pad to full batch size
        pad_size = batch_size - len(final_batch)
        padded_batch = np.pad(final_batch, ((0, pad_size), (0, 0)), mode='constant')
        batch = tf.convert_to_tensor(padded_batch, dtype=tf.float32)
        # Only keep predictions for actual samples
        predictions[start_idx:] = model_predict_fn(model, batch).numpy()[:len(final_batch)].ravel()
    
    return predictions

class CustomCheckpoint(tf.keras.callbacks.Callback):
    """Custom checkpoint callback that handles unsupported signatures"""
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.save_best_only:
            if current < self.best:
                if self.verbose:
                    print(f'\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}')
                self.best = current
                # Save weights only instead of full model
                self.model.save_weights(f"{self.filepath}_weights", save_format='tf')
        else:
            if self.verbose:
                print(f'\nEpoch {epoch}: saving model weights')
            self.model.save_weights(f"{self.filepath}_epoch_{epoch}_weights", save_format='tf')

import tensorflow as tf

# Configure for parallel execution
PARALLEL_INDIVIDUALS = 20  # Number of individuals to evaluate in parallel
BATCH_SIZE = 4096  # Increased batch size for better GPU utilization

# Add cached predict function
@tf.function(reduce_retracing=True)
def cached_predict(model, data):
    """Cached prediction function to avoid retracing"""
    return model(data, training=False)

def get_predictions(model, X, batch_size=4096):
    """Get predictions with minimal retracing"""
    # Convert input to tensor once
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    
    # Calculate number of batches
    n_samples = X_tensor.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Pre-allocate output array
    predictions = np.zeros(n_samples)
    
    # Process in batches
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = X_tensor[start_idx:end_idx]
        predictions[start_idx:end_idx] = cached_predict(model, batch).numpy().ravel()
    
    return predictions

def parallel_predict(model, X_batch):
    """XLA-optimized prediction function"""
    return model(X_batch, training=False)

def create_model(individual, X_sample):
    """Create model from individual's genome"""
    X_encoded = encode_features(individual, X_sample)
    input_shape = X_encoded.shape[1] if X_encoded is not None else 1
    return build_model(input_shape=(input_shape,), feature_cols=None)

def cached_predict(model, data):
    """XLA-optimized prediction function with GPU support"""
    with tf.device('/GPU:0'):
        return model(tf.cast(data, tf.float32), training=False)

# Add global GPU strategy setup
gpu_strategy = tf.distribute.OneDeviceStrategy("/GPU:0")

def global_predict_fn(model, inputs):
    """Global prediction function to avoid retracing"""
    return model(inputs, training=False)

def get_predictions(model, X, batch_size=4096):
    """Get predictions with minimal retracing using global function"""
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    predictions_list = []
    
    # Process in batches
    for i in range(0, len(X), batch_size):
        batch = X_tensor[i:i + batch_size]
        with tf.device('/GPU:0'):
            pred_batch = global_predict_fn(model, batch)
            predictions_list.append(pred_batch.numpy())
    
    # Concatenate all predictions and ensure 1D array
    predictions = np.concatenate(predictions_list, axis=0)
    return predictions.ravel()

def safe_convert_input_data(X, y=None):
    """Safely convert input data to proper format"""
    # Handle X
    if isinstance(X, tuple):
        # If tuple, take first element and ensure it's 2D
        X_data = X[0]
        if isinstance(X_data, pd.DataFrame):
            return X_data, y
        X_data = np.asarray(X_data)
        if X_data.ndim > 2:
            X_data = X_data.reshape(X_data.shape[0], -1)
        X_data = pd.DataFrame(X_data)
    elif isinstance(X, np.ndarray):
        if X.ndim > 2:
            X_data = X.reshape(X.shape[0], -1)
        else:
            X_data = X
        X_data = pd.DataFrame(X_data)
    elif isinstance(X, pd.DataFrame):
        X_data = X
    else:
        raise ValueError(f"Unsupported input type for X: {type(X)}")

    # Handle y if provided
    if y is not None:
        if isinstance(y, (list, np.ndarray)):
            y_data = pd.Series(y)
        elif isinstance(y, pd.Series):
            y_data = y
        else:
            raise ValueError(f"Unsupported input type for y: {type(y)}")
        return X_data, y_data
        
    return X_data

def parallel_fitness_evaluation(individuals, X, y, batch_size=BATCH_SIZE):
    """Evaluate multiple individuals in parallel with forced GPU usage"""
    try:
        # Convert input data safely
        X_data = safe_convert_input_data(X)
        
        # Rest of the function remains the same
        results = []
    
        # Process individuals in batches
        for i in range(0, len(individuals), PARALLEL_INDIVIDUALS):
            batch_inds = individuals[i:i + PARALLEL_INDIVIDUALS]
            
            # Process features
            batch_features = []
            for ind in batch_inds:
                try:
                    features = encode_features(ind, X_data)
                    if features is None:
                        batch_features.append(None)
                        continue
                        
                    features_df = pd.DataFrame(features, index=X_data.index)
                    features_df = normalize_features(features_df)
                    batch_features.append(features_df)
                except Exception as e:
                    print(f"Warning: Feature processing failed - {str(e)}")
                    batch_features.append(None)
            
            # Create models and train inside strategy scope
            try:
                with gpu_strategy.scope():
                    # Create models for valid features
                    models = []
                    valid_inds = []
                    valid_features = []
                    
                    for ind, features_df in zip(batch_inds, batch_features):
                        if features_df is not None:
                            try:
                                features = features_df.values.astype(np.float32)
                                model = build_model(input_shape=(features.shape[1],), feature_cols=None)
                                model.compile(
                                    optimizer=tf.keras.optimizers.AdamW(
                                        learning_rate=5e-4,
                                        weight_decay=1e-4,
                                        clipnorm=1.0
                                    ),
                                    loss=tf.keras.losses.Huber(delta=1.0),
                                    metrics=['mae'],
                                    jit_compile=True
                                )
                                models.append(model)
                                valid_inds.append(ind)
                                valid_features.append(features)
                            except Exception as e:
                                print(f"Error creating model: {str(e)}")
                                continue
                    
                    # Train valid models
                    for model, ind, features in zip(models, valid_inds, valid_features):
                        try:
                            X_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
                            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                            
                            # Create optimized dataset
                            dataset = create_dataset(features, y, batch_size=batch_size)
                            
                            history = model.fit(
                                dataset,
                                epochs=1,
                                verbose=10,
                                workers=1,
                                use_multiprocessing=False
                            )
                            
                            predictions = get_predictions(model, features)
                            mse = tf.reduce_mean(tf.square(predictions - y_tensor))
                            results.append((float(mse), predictions))
                            
                        except Exception as e:
                            print(f"Error training model: {str(e)}")
                            results.append((float('inf'), None))
                    
                    # Add failures
                    for ind in batch_inds:
                        if ind not in valid_inds:
                            results.append((float('inf'), None))
                            
            except Exception as e:
                print(f"Error in GPU strategy scope: {str(e)}")
                # Add infinite fitness for all individuals in batch
                results.extend([(float('inf'), None)] * len(batch_inds))
            finally:
                # Clear GPU memory
                tf.keras.backend.clear_session()
        
        return results

    except Exception as e:
        print(f"Error in parallel fitness evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return [(float('inf'), None)] * len(individuals)

