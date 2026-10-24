import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
import numpy as np

@register_keras_serializable(package='SuperNN')
class RBFLayer(Layer):
    """
    Layer of Gaussian RBF units for neural network.
    
    Arguments:
        units: Positive integer, number of output units
        gamma: RBF kernel parameter (float)
        name: Name of the layer
    """
    def __init__(self, units, gamma=1.0, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = gamma
        
    def build(self, input_shape):
        # Ensure we handle input shape correctly
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        
        # Extract feature dimension
        feature_dim = int(input_shape[-1])
        
        # Initialize centers with a smaller range for better stability
        self.mu = self.add_weight(
            name='mu',
            shape=(feature_dim, self.units),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1),
            trainable=True
        )
        
        # Initialize gamma as variable for easier tuning
        self.gamma_var = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer=tf.constant_initializer(self.gamma),
            trainable=False  # Keep gamma fixed for stability
        )
        
        super(RBFLayer, self).build(input_shape)
        
    def call(self, inputs):
        """Apply RBF transformation to inputs"""
        try:
            # Get input shape information
            batch_size = tf.shape(inputs)[0]
            feature_dim = tf.shape(inputs)[1]
            
            # Cast inputs to float32 for stability
            x = tf.cast(inputs, tf.float32)
            
            # Manual shape definition for better control
            x_expanded = tf.reshape(x, [batch_size, 1, feature_dim])
            centers = tf.reshape(tf.transpose(self.mu), [1, self.units, feature_dim])
            
            # Calculate distances with safe operations
            diff = x_expanded - centers
            squared_diff = tf.square(diff)
            squared_dist = tf.reduce_sum(squared_diff, axis=2)
            
            # Clip values for stability
            squared_dist = tf.clip_by_value(squared_dist, 0.0, 50.0)
            
            # Apply RBF kernel with explicit shape
            gamma = tf.reshape(self.gamma_var, [1])  # Ensure scalar shape
            rbf_output = tf.exp(-gamma * squared_dist)
            
            # Force shape to be fully defined
            output_shape = tf.stack([batch_size, self.units])
            rbf_output = tf.reshape(rbf_output, output_shape)
            
            return rbf_output
            
        except Exception as e:
            # Log error and return a placeholder tensor with explicit shape
            tf.print(f"Error in RBFLayer: {e}")
            # Create zero tensor with explicit shape
            return tf.zeros([tf.shape(inputs)[0], self.units], dtype=tf.float32)
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
        
    def get_config(self):
        config = super(RBFLayer, self).get_config()
        config.update({
            'units': self.units,
            'gamma': self.gamma
        })
        return config