import os
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout, BatchNormalization, Concatenate, concatenate
from tensorflow.keras.models import Model
from utils.data_manager import get_data, features, target
from utils.math import safe_rmsle

# Add imports for new layers
from tensorflow.keras.layers import (
    MultiHeadAttention, LayerNormalization, Embedding, 
    GlobalAveragePooling1D, Reshape, Conv1D, multiply,
    LeakyReLU, PReLU, Add
)
from .layers import RBFLayer  # Add this import at the top
from tensorflow.keras.saving import register_keras_serializable

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and ERROR messages

@register_keras_serializable(package='SuperNN')
class RMSLELoss(tf.keras.losses.Loss):
    def __init__(self, name='rmsle_loss', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        # Safety clipping to prevent NaNs
        y_pred = tf.clip_by_value(y_pred, -20.0, 20.0)  # Clip log space values
        
        # Ensure we have valid values for log1p
        y_pred_safe = tf.maximum(tf.exp(y_pred) - 1.0, 0.0)
        y_true_safe = tf.maximum(tf.exp(y_true) - 1.0, 0.0)
        
        # Calculate RMSLE on original scale
        squared_log_diff = tf.square(tf.math.log1p(y_pred_safe) - tf.math.log1p(y_true_safe))
        
        # Apply sample weights if provided
        if sample_weight is not None:
            squared_log_diff = squared_log_diff * sample_weight
            return tf.sqrt(tf.reduce_mean(squared_log_diff) / (tf.reduce_mean(sample_weight) + 1e-7))
        else:
            return tf.sqrt(tf.reduce_mean(squared_log_diff) + 1e-7)
            
    def get_config(self):
        return super().get_config()

@register_keras_serializable(package='SuperNN')
class RMSLEMetric(tf.keras.metrics.Metric):
    def __init__(self, name='rmsle', **kwargs):
        super().__init__(name=name, **kwargs)
        self.squared_sum = self.add_weight(name='squared_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate squared log differences
        squared_log_diff = tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))
        
        # Apply sample weights if provided
        if sample_weight is not None:
            squared_log_diff = squared_log_diff * sample_weight
            self.squared_sum.assign_add(tf.reduce_sum(squared_log_diff))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.squared_sum.assign_add(tf.reduce_sum(squared_log_diff))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
            
    def result(self):
        return tf.sqrt(self.squared_sum / (self.count + 1e-7))
        
    def reset_state(self):
        self.squared_sum.assign(0.)
        self.count.assign(0.)

@register_keras_serializable(package='SuperNN')
class ActuarialLayer(tf.keras.layers.Layer):
    """Custom layer for actuarial calculations"""
    def __init__(self, units, **kwargs):
        super(ActuarialLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        # Base risk factors
        self.base_factors = self.add_weight(
            name='base_factors',
            shape=(input_shape[-1], self.units),
            initializer='glorot_normal',
            trainable=True
        )
        # Exponential risk multipliers (for age, health conditions)
        self.exp_multipliers = self.add_weight(
            name='exp_multipliers',
            shape=(input_shape[-1], self.units),
            initializer='glorot_normal',
            trainable=True
        )
        # Logarithmic factors (for income, credit score)
        self.log_factors = self.add_weight(
            name='log_factors',
            shape=(input_shape[-1], self.units),
            initializer='glorot_normal',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(ActuarialLayer, self).build(input_shape)
        
    def call(self, inputs):
        # Linear component (base rating)
        linear = tf.matmul(inputs, self.base_factors)
        
        # Exponential component (for high-risk factors)
        exp_component = tf.matmul(
            tf.math.sigmoid(inputs), 
            self.exp_multipliers
        )
        exp_output = tf.exp(exp_component) - 1.0
        
        # Logarithmic component (for stabilizing factors)
        log_input = tf.clip_by_value(tf.abs(inputs) + 1.0, 1e-6, 1e6)
        log_component = tf.matmul(tf.math.log(log_input), self.log_factors)
        
        # Combine components using actuarial principles
        output = linear + 0.1 * exp_output + 0.2 * log_component + self.bias
        return output
    
    def get_config(self):
        config = super(ActuarialLayer, self).get_config()
        config.update({'units': self.units})
        return config

@register_keras_serializable(package='SuperNN')
class QuantileLoss(tf.keras.losses.Loss):
    """
    Quantile loss function for quantile regression
    """
    def __init__(self, quantile=0.5, name='quantile_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile = quantile
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        # Calculate quantile loss
        error = y_true - y_pred
        loss = tf.maximum(self.quantile * error, (self.quantile - 1) * error)
        
        # Apply sample weights if provided
        if sample_weight is not None:
            loss = loss * sample_weight
            return tf.reduce_mean(loss) / (tf.reduce_mean(sample_weight) + 1e-7)
        else:
            return tf.reduce_mean(loss)
            
    def get_config(self):
        config = super().get_config()
        config.update({'quantile': self.quantile})
        return config

class SuperNN:
    ARCHITECTURES = {
        0: "RBF-SVR",           # RBF kernel-based SVR emulation
        1: "Deep-Cross",        # Deep & Cross Network
        2: "TabNet-Residual",   # TabNet with residual connections
        3: "TabNet-Attention",  # TabNet with enhanced attention mechanism
        4: "GBM-Emulation",    # Neural network that emulates GBM behavior
        5: "Actuarial-Net",     # New: Actuarial network for insurance
        6: "Premium-Transformer" # New: Transformer-based insurance model
    }

    def __init__(self, input_dim, num_combinations, architecture_id=0):
        self.input_dim = input_dim
        self.num_combinations = num_combinations
        self.model = None
        self.feature_combinations = []
        self.validation_score = None
        self.model_id = None
        self.architecture_id = architecture_id % len(self.ARCHITECTURES)
        print(f"Using architecture: {self.ARCHITECTURES[self.architecture_id]}")

    def build_model(self):
        """Build model with dimension checking and input validation"""
        # Log dimensions for debugging
        print(f"Building model with dimensions: Input={self.input_dim}, Combinations={self.num_combinations}")
        
        # Define inputs with explicit shapes
        input_original = Input(shape=(self.input_dim,), name='input_original')
        input_combined = Input(shape=(self.num_combinations,), name='input_combined')
        
        # Extract important features with safe indexing and explicit shapes
        health_score = Lambda(lambda x: tf.cond(
            tf.shape(x)[1] > 3,
            lambda: x[:, 3:4],  # If index 3 exists, extract it
            lambda: tf.ones_like(x[:, :1]) * 0.5  # Otherwise use default value
        ), name='health_score_extract', output_shape=(1,))(input_original)
        
        age_factor = Lambda(lambda x: x[:, 0:1], name='age_extract', 
                           output_shape=(1,))(input_original)
        
        claims = Lambda(lambda x: tf.cond(
            tf.shape(x)[1] > 9,
            lambda: x[:, 9:10],
            lambda: tf.zeros_like(x[:, :1])
        ), name='claims_extract', output_shape=(1,))(input_original)
        
        if self.architecture_id == 0:  # RBF-SVR
            # Ensure all shapes are explicitly defined
            x1 = Dense(64, activation='relu', name='orig_dense1')(input_original)
            x1 = BatchNormalization(name='orig_bn1')(x1)
            # Add proper shape definition before RBF layer
            x1 = Dense(32, activation='relu', name='pre_rbf')(x1)  # Ensure shape is defined
            x1 = RBFLayer(32, gamma=0.1, name='orig_rbf1')(x1)
            x1 = Dense(32, activation='relu', name='orig_dense2')(x1)
            
            # Special pathway for health score with proper shape handling
            h1 = Dense(16, activation='relu', name='health_dense1')(health_score)
            # Add an intermediate layer to ensure proper shape
            h1 = Dense(8, activation='relu', name='health_pre_rbf')(h1)
            h1 = RBFLayer(8, gamma=0.1, name='health_rbf')(h1)
            
            # Special pathway for claims
            c1 = Dense(16, activation='relu', name='claims_dense1')(claims)
            # Add an intermediate layer to ensure proper shape
            c1 = Dense(8, activation='relu', name='claims_pre_rbf')(c1)
            c1 = RBFLayer(8, gamma=0.1, name='claims_rbf')(c1)
            
            # Process combined features with explicit shapes
            x2 = Dense(32, activation='relu', name='comb_dense1')(input_combined)
            x2 = BatchNormalization(name='comb_bn1')(x2)
            # Add intermediate layer to ensure proper shape
            x2 = Dense(16, activation='relu', name='comb_pre_rbf')(x2)
            x2 = RBFLayer(16, gamma=0.1, name='comb_rbf1')(x2)
            x2 = Dense(16, activation='relu', name='comb_dense2')(x2)
            
            # Use safer concatenation with explicit shapes
            try:
                # Ensure all tensors have defined shapes
                x1_shape = x1.shape[1] or 32
                x2_shape = x2.shape[1] or 16
                h1_shape = h1.shape[1] or 8
                c1_shape = c1.shape[1] or 8
                
                # Force shapes to be defined before concatenation
                x1 = Dense(x1_shape, name='x1_reshape')(x1)
                x2 = Dense(x2_shape, name='x2_reshape')(x2)
                h1 = Dense(h1_shape, name='h1_reshape')(h1)
                c1 = Dense(c1_shape, name='c1_reshape')(c1)
                
                # Now concatenate with defined shapes
                combined = concatenate([x1, x2, h1, c1], name='concat')
            except Exception as e:
                # Fallback for concatenation errors
                print(f"Warning: Using fallback concatenation: {e}")
                # Use fixed size dense layers for each branch
                x1 = Dense(32, activation='relu')(x1)
                x2 = Dense(16, activation='relu')(x2)
                h1 = Dense(8, activation='relu')(h1)
                c1 = Dense(8, activation='relu')(c1)
                combined = concatenate([x1, x2, h1, c1], name='concat_safe')
            
            x = Dense(64, activation='relu', name='final_dense1')(combined)
            x = BatchNormalization(name='final_bn')(x)
        
        elif self.architecture_id == 1:  # Cross-Feature Network - FIXED VERSION
            # Process HealthScore separately with simpler architecture
            h1 = Dense(16, activation='relu', name='health_dense1')(health_score)
            h1 = BatchNormalization(name='health_bn1')(h1)
            
            # Simplified architecture for original features
            x1 = Dense(32, activation='relu', name='orig_dense1')(input_original)
            x1 = BatchNormalization(name='orig_bn1')(x1)
            x1 = Dense(32, activation='relu', name='orig_dense2')(x1)
            x1 = BatchNormalization(name='orig_bn2')(x1)
            
            # Process combined features with minimal layers
            x2 = Dense(16, activation='relu', name='comb_dense1')(input_combined)
            x2 = BatchNormalization(name='comb_bn1')(x2)
            
            # Combine all features
            combined = concatenate([x1, x2, h1], name='concat')
            x = Dense(32, activation='relu', name='final_dense1')(combined)
            x = BatchNormalization(name='final_bn1')(x)
            x = Dense(16, activation='relu', name='final_dense2')(x)
            
        elif self.architecture_id == 2:  # TabNet-Residual
            def residual_feature_selection(x, units, name_prefix):
                shortcut = x
                
                # Feature selection path with insurance-specific attention
                attention = Dense(x.shape[-1], activation='sigmoid', name=f'{name_prefix}_attention')(x)
                x = multiply([x, attention], name=f'{name_prefix}_attended')
                x = Dense(units, activation='selu', name=f'{name_prefix}_dense')(x)
                x = BatchNormalization(name=f'{name_prefix}_bn')(x)
                x = Dropout(0.1, name=f'{name_prefix}_dropout')(x)
                
                # Residual connection
                if shortcut.shape[-1] != units:
                    shortcut = Dense(units, name=f'{name_prefix}_shortcut')(shortcut)
                
                x = Add(name=f'{name_prefix}_add')([x, shortcut])
                return x
            
            # Process original features with residual blocks
            x1 = residual_feature_selection(input_original, 64, 'orig1')
            x1 = residual_feature_selection(x1, 64, 'orig2')
            
            # Special pathway for health score
            h1 = Dense(32, activation='selu')(health_score)
            h1 = BatchNormalization()(h1)
            
            # Special pathway for age
            a1 = Dense(32, activation='selu')(age_factor)
            a1 = BatchNormalization()(a1)
            
            # Process combined features
            x2 = residual_feature_selection(input_combined, 32, 'comb1')
            x2 = residual_feature_selection(x2, 32, 'comb2')
            
            combined = concatenate([x1, x2, h1, a1])
            x = Dense(96, activation='selu')(combined)
            x = BatchNormalization()(x)
            x = Dropout(0.15)(x)
            
        elif self.architecture_id == 3:  # TabNet-Attention
            def attention_feature_selection(x, units):
                # Reshape for self-attention
                x_reshaped = Reshape((1, -1))(x)
                
                # Self-attention mechanism
                attention = MultiHeadAttention(
                    num_heads=4, key_dim=16
                )(x_reshaped, x_reshaped)
                attention = Reshape((-1,))(attention)
                
                # Feature selection gate
                gate = Dense(x.shape[-1], activation='sigmoid')(attention)
                x = x * gate
                
                x = Dense(units, activation='gelu')(x)
                x = BatchNormalization()(x)
                return x
            
            x1 = attention_feature_selection(input_original, 64)
            x2 = attention_feature_selection(input_combined, 32)
            
            combined = concatenate([x1, x2])
            x = Dense(96, activation='gelu')(combined)
            x = BatchNormalization()(x)
            
        elif self.architecture_id == 4:  # GBM-Emulation
            def gbm_block(x, units, name_prefix):
                # Multiple decision trees emulation with more trees
                trees = []
                for i in range(8):  # More trees for better coverage
                    tree = Dense(units, activation='relu', name=f'{name_prefix}_tree{i}_l1')(x)
                    tree = BatchNormalization(name=f'{name_prefix}_tree{i}_bn1')(tree)
                    tree = Dense(units//2, activation='relu', name=f'{name_prefix}_tree{i}_l2')(tree)
                    tree = BatchNormalization(name=f'{name_prefix}_tree{i}_bn2')(tree)
                    trees.append(tree)
                
                # Combine trees (like boosting)
                x = concatenate(trees, name=f'{name_prefix}_forest')
                x = Dense(units, activation='gelu', name=f'{name_prefix}_aggregate')(x)
                x = BatchNormalization(name=f'{name_prefix}_bn_final')(x)
                return x
            
            # Process original features with GBM blocks
            x1 = gbm_block(input_original, 64, 'orig')
            
            # Special pathway for health score
            h1 = Dense(16, activation='relu')(health_score)
            h1 = BatchNormalization()(h1)
            
            # Process combined features
            x2 = gbm_block(input_combined, 32, 'comb')
            
            combined = concatenate([x1, x2, h1])
            x = Dense(128, activation='gelu')(combined)
            x = BatchNormalization()(x)
            
        elif self.architecture_id == 5:  # Actuarial-Net
            # Base risk assessment
            x1 = ActuarialLayer(64, name='actuarial_base')(input_original)
            x1 = BatchNormalization(name='base_bn')(x1)
            x1 = PReLU(name='base_activation')(x1)
            
            # Risk factor pathway
            risk = Dense(32, activation='selu', name='risk_dense1')(input_original)
            risk = BatchNormalization(name='risk_bn1')(risk)
            risk = ActuarialLayer(32, name='risk_actuarial')(risk)
            risk = PReLU(name='risk_activation')(risk)
            
            # Health score gets special treatment in actuarial models
            health_path = Dense(16, activation='selu', name='health_dense1')(health_score)
            health_path = BatchNormalization(name='health_bn1')(health_path)
            # Exponential risk factor for health
            health_factor = Dense(1, activation='sigmoid', name='health_factor')(health_score)
            health_impact = Lambda(lambda x: tf.exp(3.0 * (1.0 - x)), name='health_impact')(health_factor)
            
            # Age gets special treatment in actuarial models
            age_path = Dense(16, activation='selu', name='age_dense1')(age_factor)
            age_path = BatchNormalization(name='age_bn1')(age_path)
            # Exponential risk factor for age
            age_factor_out = Dense(1, activation='sigmoid', name='age_factor')(age_factor)
            age_impact = Lambda(lambda x: tf.exp(2.0 * x), name='age_impact')(age_factor_out)
            
            # Claims history pathway
            claims_path = Dense(16, activation='selu', name='claims_dense1')(claims)
            claims_path = BatchNormalization(name='claims_bn1')(claims_path)
            # Linear risk factor for claims
            claims_impact = Dense(1, activation='relu', name='claims_impact')(claims)
            
            # Combine risk factors
            combined_risk = multiply([health_impact, age_impact, claims_impact], name='risk_multiplication')
            combined_risk = Lambda(lambda x: 0.1 + 0.9 * x, name='risk_scaling')(combined_risk)
            
            # Process combined features for additional risk factors
            x2 = Dense(32, activation='selu', name='comb_dense1')(input_combined)
            x2 = BatchNormalization(name='comb_bn1')(x2)
            x2 = Dense(32, activation='selu', name='comb_dense2')(x2)
            
            # Combine all pathways
            features_combined = concatenate([x1, risk, health_path, age_path, claims_path, x2], name='all_features')
            features_combined = Dense(64, activation='selu', name='features_dense')(features_combined)
            
            # Final premium calculation layers
            x = Dense(32, activation='selu', name='premium_dense1')(features_combined)
            x = BatchNormalization(name='premium_bn1')(x)
            x = Dense(16, activation='selu', name='premium_dense2')(x)
            # Apply the risk multiplier
            x = Lambda(lambda inputs: inputs[0] * inputs[1], name='risk_adjustment')([x, combined_risk])
            
        elif self.architecture_id == 6:  # Premium-Transformer
            # Transformer-based architecture for insurance premium estimation
            
            # Project features to embedding dimension
            embed_dim = 64
            x1 = Dense(embed_dim, name='orig_projection')(input_original)
            x1 = Reshape((1, embed_dim), name='orig_reshape')(x1)
            
            x2 = Dense(embed_dim, name='comb_projection')(input_combined)
            x2 = Reshape((1, embed_dim), name='comb_reshape')(x2)
            
            # Special tokens for important features
            health_embed = Dense(embed_dim, name='health_projection')(health_score)
            health_embed = Reshape((1, embed_dim), name='health_reshape')(health_embed)
            
            age_embed = Dense(embed_dim, name='age_projection')(age_factor)
            age_embed = Reshape((1, embed_dim), name='age_reshape')(age_embed)
            
            claims_embed = Dense(embed_dim, name='claims_projection')(claims)
            claims_embed = Reshape((1, embed_dim), name='claims_reshape')(claims_embed)
            
            # Combine all tokens
            tokens = concatenate([x1, x2, health_embed, age_embed, claims_embed], axis=1, name='token_concat')
            
            # Add positional embeddings
            pos_embedding = self.add_positional_embedding(tokens, name='pos_embedding')
            x = tokens + pos_embedding
            
            # Apply transformer layers
            for i in range(3):
                # Multi-head attention
                attention_output = MultiHeadAttention(
                    num_heads=4,
                    key_dim=embed_dim // 4,
                    name=f'mha_{i}'
                )(x, x)
                
                # Add & norm
                x = Add(name=f'add1_{i}')([x, attention_output])
                x = LayerNormalization(epsilon=1e-6, name=f'ln1_{i}')(x)
                
                # Feed-forward network
                ffn = Dense(embed_dim * 2, activation='gelu', name=f'ffn1_{i}')(x)
                ffn = Dropout(0.1, name=f'drop1_{i}')(ffn)
                ffn = Dense(embed_dim, name=f'ffn2_{i}')(ffn)
                
                # Add & norm
                x = Add(name=f'add2_{i}')([x, ffn])
                x = LayerNormalization(epsilon=1e-6, name=f'ln2_{i}')(x)
            
            # Global pooling
            x = GlobalAveragePooling1D(name='gap')(x)
            x = BatchNormalization(name='final_bn')(x)
            x = Dense(32, activation='gelu', name='final_dense')(x)
        
        # Final common layers for all architectures with explicit shapes
        x = Dropout(0.25)(x)
        # Add a layer to force shape definition
        x = Dense(16, activation='gelu', kernel_initializer='he_normal')(x)
        output = Dense(1, activation='linear')(x)
        
        # Create model with explicit input/output definition
        self.model = Model(inputs=[input_original, input_combined], outputs=output)

    def add_positional_embedding(self, x, name='positional_embedding'):
        """Add positional embeddings to the input tokens"""
        seq_length = tf.shape(x)[1]
        embed_dim = tf.shape(x)[2]
        
        # Create positional embeddings
        position_indices = tf.range(seq_length, dtype=tf.float32)
        position_indices = tf.expand_dims(position_indices, axis=1)
        
        # Sinusoidal position encoding
        div_term = tf.exp(tf.range(0, embed_dim, 2, dtype=tf.float32) * (-np.log(10000.0) / embed_dim))
        div_term = tf.expand_dims(div_term, axis=0)
        
        sin_terms = tf.sin(position_indices * div_term)
        cos_terms = tf.cos(position_indices * div_term)
        
        # Combine sin and cos terms
        sin_indices = tf.range(0, embed_dim, 2, dtype=tf.int32)
        cos_indices = tf.range(1, embed_dim, 2, dtype=tf.int32)
        
        pos_encoding = tf.zeros([seq_length, embed_dim], dtype=tf.float32)
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.tile(tf.expand_dims(tf.range(seq_length), 1), [1, tf.shape(sin_indices)[0]]),
                      tf.tile(tf.expand_dims(sin_indices, 0), [seq_length, 1])], axis=2),
            tf.reshape(sin_terms, [-1])
        )
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.tile(tf.expand_dims(tf.range(seq_length), 1), [1, tf.shape(cos_indices)[0]]),
                      tf.tile(tf.expand_dims(cos_indices, 0), [seq_length, 1])], axis=2),
            tf.reshape(cos_terms, [-1])
        )
        
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)
        return pos_encoding

    def compile_model(self, learning_rate=0.001, metrics=['mae'], loss=None, quantile=None):
        """Enhanced compile method with insurance-specific adjustments"""
        # Adjust learning rate for problematic architectures
        if self.architecture_id == 1:  # Deep-Cross - reduce learning rate
            learning_rate = learning_rate * 0.5
            print(f"Using reduced learning rate for Deep-Cross: {learning_rate}")
        
        # Use AdamW with weight decay for better generalization
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,  # Add weight decay
            clipnorm=0.25       # Reduce gradient clipping for stability
        )
        
        # Add insurance-specific metrics
        all_metrics = metrics.copy() if isinstance(metrics, list) else [metrics]
        all_metrics.append(RMSLEMetric())  # Always include RMSLE
        
        # Set appropriate loss function
        if loss is None:
            if quantile is not None:
                loss = QuantileLoss(quantile=quantile)
                print(f"Using quantile loss with q={quantile}")
            else:
                loss = tf.keras.losses.Huber(delta=1.0)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=all_metrics
        )

    def set_feature_combinations(self, combinations, model_id):
        """Store the feature combinations and model ID"""
        if not isinstance(combinations, (list, tuple)):
            raise ValueError("combinations must be a list or tuple")
        self.feature_combinations = list(combinations)  # Ensure it's a list
        self.model_id = model_id
        print(f"Set {len(self.feature_combinations)} feature combinations for model {model_id}")
        
    def train_model(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=None):
        """Train the model with improved callbacks and error handling"""
        # Adjust batch size for problematic architectures
        if self.architecture_id == 1:  # Deep-Cross - use smaller batch size
            batch_size = max(16, batch_size // 2)
            epochs = min(3, epochs)  # Limit epochs for Deep-Cross
            print(f"Using reduced batch size {batch_size} and {epochs} epochs for Deep-Cross")
        
        # Set memory limit for TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Memory growth enabled for GPU")
            except RuntimeError as e:
                print(f"Memory growth setting error: {e}")

        callbacks = callbacks or []
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,  # More aggressive early stopping
                restore_best_weights=True
            )
        ])
        
        try:
            # Validate inputs before training
            if isinstance(X_train, list):
                print(f"X_train shapes: {[x.shape for x in X_train]}")
            else:
                print(f"X_train shape: {X_train.shape}")
            print(f"y_train shape: {y_train.shape}")
            
            # Calculate target statistics
            y_train_original = np.expm1(y_train)
            target_mean = np.mean(y_train_original)
            target_std = np.std(y_train_original)
            print(f"\nTarget Statistics:")
            print(f"Mean: {target_mean:.2f}")
            print(f"Std: {target_std:.2f}")
            print(f"Min: {np.min(y_train_original):.2f}")
            print(f"Max: {np.max(y_train_original):.2f}")
            
            # Lower precision for problematic architectures
            if self.architecture_id == 1:
                print("Setting mixed precision for numerical stability")
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
            
            print("\nModel Architecture:")
            self.model.summary()
            print(f"\nTraining {self.ARCHITECTURES[self.architecture_id]} model {self.model_id}...")
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get validation predictions
            val_idx = int(len(X_train[0]) * (1 - validation_split))
            X_val = [x[val_idx:] for x in X_train]
            y_val = y_train[val_idx:]
            
            # Get predictions
            val_preds = self.predict(X_val).flatten()
            
            # Calculate RMSLE on original scale
            y_val_original = np.expm1(y_val)
            val_preds_original = np.expm1(val_preds)
            
            # Store final validation score
            self.validation_score = safe_rmsle(y_val_original, val_preds_original)
            
            print(f"\nModel {self.model_id} Training Results:")
            print("-" * 40)
            print(f"Architecture: {self.ARCHITECTURES[self.architecture_id]}")
            print(f"Input Features: {self.input_dim}")
            print(f"Feature Combinations: {self.num_combinations}")
            if 'rmsle' in history.history:
                print(f"Training RMSLE: {history.history['rmsle'][-1]:.4f}")
            print(f"Validation RMSLE: {self.validation_score:.4f}")
            print(f"Final Loss: {history.history['loss'][-1]:.4f}")
            print("-" * 40)
            
            return history
            
        except Exception as e:
            print(f"\nError during model training: {e}")
            # Set a default validation score
            self.validation_score = 999.0
            # Return empty history
            return {"loss": [999.0]}

    def evaluate_model(self, X_test, y_test):
        # Print shapes for debugging
        print(f"Evaluating model with original features shape: {X_test[0].shape}")
        print(f"Combination features shape: {X_test[1].shape}")
        
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {loss}")
        return loss
        
    def predict(self, X):
        """Predict with dimension checking and proper shape access"""
        # Verify shape compatibility
        if isinstance(X, list) and len(X) == 2:
            input_original_shape = X[0].shape[1]
            input_combined_shape = X[1].shape[1]
            
            # Print shapes for debugging
            print(f"Predicting with original features shape: {X[0].shape}")
            print(f"Combination features shape: {X[1].shape}")
            
            # Check if shapes match model expectations - FIXED to use output_shape or inputs directly
            try:
                # Try getting shapes from input layers' output_shape attribute
                model_input_shape = self.model.get_layer('input_original').output_shape[1]
                model_combined_shape = self.model.get_layer('input_combined').output_shape[1]
            except (AttributeError, IndexError):
                # Fallback to model.inputs which is more reliable
                try:
                    if len(self.model.inputs) >= 2:
                        model_input_shape = self.model.inputs[0].shape[1]
                        model_combined_shape = self.model.inputs[1].shape[1]
                    else:
                        # In case the model structure is unexpected
                        print("Warning: Could not determine expected input shapes from model")
                        # Make a best guess from X
                        model_input_shape = input_original_shape
                        model_combined_shape = input_combined_shape
                except Exception as e:
                    print(f"Error determining model input shapes: {e}")
                    # Just use provided X shapes
                    model_input_shape = input_original_shape
                    model_combined_shape = input_combined_shape
            
            if input_original_shape != model_input_shape or input_combined_shape != model_combined_shape:
                print(f"Warning: Input shape mismatch!")
                print(f"Expected shapes: Original={model_input_shape}, Combined={model_combined_shape}")
                print(f"Received shapes: Original={input_original_shape}, Combined={input_combined_shape}")
                
                # Handle shape mismatch gracefully
                if input_original_shape != model_input_shape:
                    # Create adapter
                    print("Adapting original features dimension...")
                    if input_original_shape > model_input_shape:
                        # Too many features - select subset based on importance
                        X[0] = X[0][:, :model_input_shape]
                    else:
                        # Too few features - pad with zeros
                        padding = np.zeros((X[0].shape[0], model_input_shape - input_original_shape))
                        X[0] = np.hstack([X[0], padding])
                
                if input_combined_shape != model_combined_shape:
                    # Create adapter
                    print("Adapting combined features dimension...")
                    if input_combined_shape > model_combined_shape:
                        # Too many features - select subset
                        X[1] = X[1][:, :model_combined_shape]
                    else:
                        # Too few features - pad with zeros
                        padding = np.zeros((X[1].shape[0], model_combined_shape - input_combined_shape))
                        X[1] = np.hstack([X[1], padding])
        
        # Make predictions with adapted data
        try:
            return self.model.predict(X, verbose=0)
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Return empty array with right shape as fallback
            if isinstance(X, list):
                return np.zeros((X[0].shape[0], 1))
            return np.zeros((X.shape[0], 1))

    def save_model(self, base_path='/home/USER/Projects/ikaggle/KISS/models'):
        """Save the model and its metadata"""
        # Create models directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(base_path, f'model_{self.model_id}.keras')
        self.model.save(model_path)
        
        # Convert feature combinations to list of lists for JSON serialization
        feature_combos = [[str(combo[0]), str(combo[1])] for combo in self.feature_combinations]
        
        # Save metadata with architecture info
        metadata = {
            'model_id': self.model_id,
            'architecture_id': int(self.architecture_id),  # Ensure it's saved as integer
            'architecture_name': str(self.ARCHITECTURES[self.architecture_id]),  # Save name explicitly
            'feature_combinations': feature_combos,
            'validation_score': float(self.validation_score),  # Convert to float
            'input_dim': self.input_dim,
            'num_combinations': self.num_combinations
        }
        
        metadata_path = os.path.join(base_path, f'model_{self.model_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model and metadata saved to {base_path}")

    @classmethod
    def load_model(cls, model_id, base_path='/home/USER/Projects/ikaggle/KISS/models'):
        """Load a model and its metadata with enhanced error handling"""
        # Load metadata
        metadata_path = os.path.join(base_path, f'model_{model_id}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Create instance with correct architecture_id
        instance = cls(
            input_dim=metadata['input_dim'],
            num_combinations=metadata['num_combinations'],
            architecture_id=int(metadata.get('architecture_id', 0))
        )
        instance.model_id = metadata['model_id']
        
        # Convert feature combinations
        instance.feature_combinations = [tuple(combo) for combo in metadata['feature_combinations']]
        instance.validation_score = float(metadata.get('validation_score', float('inf')))
        
        # Define custom objects for loading
        custom_objects = {
            'RMSLELoss': RMSLELoss,
            'RMSLEMetric': RMSLEMetric,
            'CrossLayer': CrossLayer,
            'HealthScoreCorrelationLayer': HealthScoreCorrelationLayer,
            'RBFLayer': RBFLayer
        }
        
        # Add any custom loss function that might have been used
        def rmsle_loss(y_true, y_pred):
            return tf.keras.backend.sqrt(tf.keras.backend.mean(
                tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))
            ))
        
        custom_objects['rmsle_loss'] = rmsle_loss
        
        # Load model with custom objects
        model_path = os.path.join(base_path, f'model_{model_id}.keras')
        try:
            instance.model = tf.keras.models.load_model(
                model_path, 
                custom_objects=custom_objects,
                compile=True
            )
            
            # Verify model has expected inputs
            if not hasattr(instance.model, 'inputs') or not instance.model.inputs:
                print(f"Warning: Model {model_id} has no inputs defined")
                
            # Test the model with dummy inputs to verify it works
            dummy_original = np.zeros((1, instance.input_dim), dtype=np.float32)
            dummy_combined = np.zeros((1, instance.num_combinations), dtype=np.float32)
            try:
                _ = instance.model.predict([dummy_original, dummy_combined], verbose=0)
                print(f"Successfully verified model {model_id} with dummy inputs")
            except Exception as e:
                print(f"Warning: Model prediction test failed: {e}")
                
            print(f"Loaded model {model_id} with architecture: {instance.ARCHITECTURES[instance.architecture_id]}")
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            print("Removing problematic model files...")
            # Remove the problematic files if they exist
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            raise RuntimeError(f"Failed to load model {model_id}. Model files have been removed.")
        
        return instance

    def summary(self):
        self.model.summary()

    def create_submission(self, X_test, submission_path=None):
        """
        Create a prediction submission file directly after training
        
        Args:
            X_test: Test data features as a list [original_features, combined_features]
            submission_path: Optional path to save the submission file
        
        Returns:
            Path to the saved submission file
        """
        # Generate predictions
        print(f"Generating predictions for model {self.model_id}...")
        try:
            preds = self.predict(X_test)
            
            # Convert predictions back to original scale (from log space)
            preds_original = np.expm1(preds).flatten()
            
            # Ensure all predictions are valid
            preds_original = np.maximum(preds_original, 0.0)
            
            # Create DataFrame with IDs and predictions
            submission = pd.DataFrame({
                'id': range(1200000, 1200000 + len(preds_original)),
                'Premium Amount': preds_original
            })
            
            # Default path if none provided
            if submission_path is None:
                os.makedirs('/home/USER/Projects/ikaggle/KISS/results', exist_ok=True)
                submission_path = f'/home/USER/Projects/ikaggle/KISS/results/submission_model_{self.model_id}.csv'
            
            # Save to file
            submission.to_csv(submission_path, index=False)
            print(f"Submission file saved to {submission_path}")
            
            # Quick data validation
            print(f"Prediction stats: Min={preds_original.min():.2f}, "
                  f"Max={preds_original.max():.2f}, "
                  f"Mean={preds_original.mean():.2f}")
            
            return submission_path
            
        except Exception as e:
            print(f"Error generating submission: {e}")
            # Create emergency submission with zeros if prediction fails
            emergency_path = f'/home/USER/Projects/ikaggle/KISS/results/emergency_submission_{self.model_id}.csv'
            ids = range(1200000, 1200000 + X_test[0].shape[0])
            pd.DataFrame({'id': ids, 'Premium Amount': [500.0] * len(ids)}).to_csv(emergency_path, index=False)
            print(f"Emergency submission created at {emergency_path}")
            return emergency_path

@register_keras_serializable(package='SuperNN')
class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, use_bias=True, **kwargs):
        super(CrossLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
    def build(self, input_shape):
        # Check if input_shape is a list (for functional API) or a single shape (for Sequential API)
        if isinstance(input_shape, list):
            # We expect two inputs for cross interaction
            if len(input_shape) != 2:
                raise ValueError(f"Expected 2 input shapes, got {len(input_shape)}")
            x0_shape = input_shape[0]
            x_shape = input_shape[1]
        else:
            # Single tensor input - split the features
            x0_shape = x_shape = input_shape
        
        # Use small initialization for numerical stability
        self.x0_projection = self.add_weight(
            name='x0_projection',
            shape=[int(x0_shape[-1]), self.output_dim],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),
            trainable=True
        )
        
        # Input projection for x
        if int(x_shape[-1]) != self.output_dim:
            self.x_projection = self.add_weight(
                name='x_projection',
                shape=[int(x_shape[-1]), self.output_dim],
                initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01),
                trainable=True
            )
        else:
            self.x_projection = None
        
        # Cross interaction weights - small values to start
        self.cross_kernel = self.add_weight(
            name='cross_kernel',
            shape=[self.output_dim, self.output_dim],
            initializer=tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005),
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[self.output_dim],
                initializer='zeros',
                trainable=True
            )
        
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.built = True
            
    def call(self, inputs):
        # Use try-except for robust error handling
        try:
            # Support both list inputs and single tensor
            if isinstance(inputs, list):
                if len(inputs) != 2:
                    raise ValueError(f"Expected 2 inputs, got {len(inputs)}")
                x0, x = inputs
            else:
                x0 = x = inputs
            
            # Project inputs (with clipping for stability)
            x0_proj = tf.clip_by_value(
                tf.matmul(x0, self.x0_projection),
                -10.0, 10.0
            )
            
            if self.x_projection is not None:
                x_proj = tf.clip_by_value(
                    tf.matmul(x, self.x_projection),
                    -10.0, 10.0
                )
            else:
                x_proj = x
            
            # Simple cross interaction with no normalization - more stable
            cross = tf.matmul(x0_proj, self.cross_kernel * 0.1)  # Scale down weights
            
            if self.use_bias:
                cross = cross + self.bias
                
            # Add rather than multiply for stability
            return x_proj + tf.clip_by_value(cross, -1.0, 1.0)
        
        except Exception as e:
            # Fallback mechanism - return the input if something goes wrong
            tf.print("Error in CrossLayer:", e)
            if isinstance(inputs, list):
                return inputs[1]  # Return second input as fallback
            return inputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.output_dim)

    def get_config(self):
        config = super(CrossLayer, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'use_bias': self.use_bias
        })
        return config
    
    # Add methods for proper serialization
    def get_build_config(self):
        return {
            'input_shape': ([None, self.input_dim], [None, self.input_dim])
        }
    
    def build_from_config(self, config):
        if not self.built:
            self.build(config['input_shape'])

@register_keras_serializable(package='SuperNN')
class HealthScoreCorrelationLayer(tf.keras.layers.Layer):
    def __init__(self, target_correlation=0.48, penalty_weight=0.1, **kwargs):
        super().__init__(**kwargs)
        self.target_correlation = target_correlation
        self.penalty_weight = penalty_weight

    def call(self, inputs):
        health_scores, predictions = inputs
        
        # Cast inputs to float32
        health_scores = tf.cast(health_scores, tf.float32)
        predictions = tf.cast(predictions, tf.float32)
        
        # Calculate means using keras operations
        h_mean = tf.keras.backend.mean(health_scores)
        p_mean = tf.keras.backend.mean(predictions)
        
        # Center the variables
        h_centered = health_scores - h_mean
        p_centered = predictions - p_mean
        
        # Calculate correlation components using keras operations
        numerator = tf.keras.backend.mean(h_centered * p_centered)
        h_std = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.square(h_centered)))
        p_std = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.square(p_centered)))
        
        # Calculate correlation
        correlation = numerator / (h_std * p_std + tf.keras.backend.epsilon())
        correlation = tf.abs(correlation)
        
        # Compute penalty
        penalty = self.penalty_weight * tf.abs(correlation - self.target_correlation)
        
        return penalty

    def get_config(self):
        config = super().get_config()
        config.update({
            'target_correlation': self.target_correlation,
            'penalty_weight': self.penalty_weight
        })
        return config

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([])

# Example usage
if __name__ == "__main__":
    # Simulated feature values (replace with actual data)
    feature_values = np.random.rand(1, len(features.split(',')))

    # Initialize and build the model
    super_nn = SuperNN(input_dim=len(features.split(',')), num_combinations=1)
    super_nn.build_model()
    super_nn.summary()

    # Example prediction
    prediction = super_nn.predict([feature_values, np.zeros((1, 1))])
    print(f"Prediction: {prediction}")