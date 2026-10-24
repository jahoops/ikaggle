import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

def build_insurance_model(input_shape, model_type='normal', learning_rate=0.001):
    """
    Build a neural network model for insurance premium prediction with XLA stability fixes
    
    Args:
        input_shape: Number of input features
        model_type: Type of model architecture ('normal', 'deep', 'wide', 'error_segment')
        learning_rate: Learning rate for the optimizer
        
    Returns:
        A compiled Keras model
    """
    # Set regularization based on model type
    if model_type == 'error_segment':
        # For the error segment, use stronger regularization
        reg = l1_l2(l1=0.01, l2=0.01)
        dropout_rate = 0.3
    else:
        reg = l1_l2(l1=0.001, l2=0.001)
        dropout_rate = 0.2
    
    # Create model architecture based on type
    model = Sequential()
    
    # Add this configuration to disable problematic XLA optimizations
    tf.config.optimizer.set_jit(False)  # Disable JIT compilation for problematic models
    
    # For the error segment model specifically, use higher precision
    if model_type == 'error_segment':
        # Force full precision for error segment model to avoid numerical issues
        tf.keras.mixed_precision.set_global_policy('float32')
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg, 
                               input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=reg))
        model.add(layers.Dense(1, activation='linear'))
        
    elif model_type == 'normal':
        # Standard architecture
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=reg, 
                               input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(16, activation='relu', kernel_regularizer=reg))
        model.add(layers.Dense(1, activation='linear')) # Linear output for regression
        
    elif model_type == 'deep':
        # Deeper architecture
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg, 
                               input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(16, activation='relu', kernel_regularizer=reg))
        model.add(layers.Dense(1, activation='linear'))
        
    elif model_type == 'wide':
        # Wider architecture
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=reg, 
                               input_shape=(input_shape,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(128, activation='relu', kernel_regularizer=reg))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1, activation='linear'))
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def build_ensemble_model(models, input_shape):
    """
    Build an ensemble model that combines multiple base models.
    
    Args:
        models: List of trained base models
        input_shape: Input shape for the models
    
    Returns:
        A compiled Keras ensemble model
    """
    # Create input layer
    inputs = tf.keras.Input(shape=(input_shape,))
    
    # Get predictions from each model
    model_outputs = [model(inputs) for model in models]
    
    # Average the predictions
    ensemble_output = layers.Average()(model_outputs)
    
    # Create and compile the ensemble model
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=ensemble_output)
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return ensemble_model
