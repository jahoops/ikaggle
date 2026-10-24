import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class RMSLELoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        # Check for NaNs and Infs in y_true and y_pred
        tf.debugging.check_numerics(y_true, message='Checking for NaNs/Infs in y_true')
        tf.debugging.check_numerics(y_pred, message='Checking for NaNs/Infs in y_pred')
        
        # Ensure y_pred is non-negative
        y_pred = tf.maximum(y_pred, 0)
        
        first_log = tf.math.log1p(y_pred)
        second_log = tf.math.log1p(y_true)
        
        # Check for NaNs and Infs in the log-transformed values
        tf.debugging.check_numerics(first_log, message='Checking for NaNs/Infs in first_log')
        tf.debugging.check_numerics(second_log, message='Checking for NaNs/Infs in second_log')
        
        return tf.sqrt(tf.reduce_mean(tf.square(first_log - second_log)))

class PremiumAmountPredictor:
    def __init__(self, input_dim):
        self.model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    
    def compile_model(self, learning_rate=0.001, metrics=None):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=RMSLELoss(), metrics=metrics)
    
    def train_model(self, X_train, y_train, epochs=10, batch_size=128, callbacks=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)
    
    def evaluate_model(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        return self.model.predict(X)
