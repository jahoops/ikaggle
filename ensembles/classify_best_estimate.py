import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
import tensorflow as tf  # Import TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
import logging

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(merged_train, train_bestof):
    runClassification(merged_train, train_bestof)

def runClassification(merged_train, train_bestof):
    # Merge the closest_model from train_bestof into merged_train
    data = merged_train.merge(train_bestof[['id', 'closest_model']], on='id')
    
    # Define the features and target
    features = [
        'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 'PreviousClaims', 'VehicleAge', 
        'CreditScore', 'InsuranceDuration', 'SignupYear', 'SignupMonth', 'MaritalStatus_Married', 
        'EducationLevel_PhD', 'Occupation_SelfEmployed', 'Occupation_nan', 'Location_Rural', 
        'Location_Urban', 'CustomerFeedback_Average', 'CustomerFeedback_Good', 'SmokingStatus_No', 
        'ExerciseFrequency_Daily', 'PropertyType_Apartment', 'PropertyType_House', 'avg_pred', 
        'median_pred', 'catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 
        'randomforest', 'ridge', 'svr', 'xgboost'
    ]
    target = 'closest_model'
    
    # Split data into features and target
    X = data[features]
    y = data[target]
    
    # Encode the target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Define the neural network model
    def create_model():
        model = Sequential([
            tf.keras.Input(shape=(X.shape[1],)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')  # Use softmax activation for multi-class classification
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Set up callbacks for checkpointing and early stopping
    checkpoint_path = 'ensembles/intermediate/classifier_model_checkpoint.h5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    # Train the model
    model = create_model()
    logger.info("Starting model training...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, verbose=1, callbacks=[checkpoint, early_stopping])
    
    # Load the best model
    model.load_weights(checkpoint_path)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_val, y_pred_classes)
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save the final model
    final_model_path = 'ensembles/intermediate/classifier_model.h5'
    model.save(final_model_path)
    logger.info(f"Model saved to {final_model_path}")

    # Print the current score
    print(f"Validation Accuracy: {accuracy:.4f}")

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")
    merged_train = pd.read_csv(data_dir / "merged_train.csv")
    train_bestof = pd.read_csv(data_dir / "train_bestof.csv")
    start(merged_train, train_bestof)
