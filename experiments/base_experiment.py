from abc import ABC, abstractmethod
import json
from pathlib import Path
import pickle
import time
from experiments.data_manager import data_manager  # Updated import
import logging
import traceback
from datetime import datetime
import pandas as pd
import os
import numpy as np

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, output_dir: Path, test_mode: bool = False):  # Added test_mode parameter
        self.output_dir = Path(output_dir)
        self.test_mode = test_mode  # Stored as an instance attribute
        self.logger = logging.getLogger(self.__class__.__name__)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_time(self, timestamp): 
        # Convert the timestamp to a datetime object 
        dt = datetime.fromtimestamp(timestamp) 
        # Format the datetime object 
        return dt.strftime("%Y%m%d%H%M")  
         
    def dropIdPremium(self, data):
        """Drop ID and PremiumAmount columns consistently."""
        return data.drop(columns=['id', 'PremiumAmount'])

    def calculate_confidence(self, model, X):
        """Calculate meaningful confidence scores"""
        predictions = []
        # Get predictions from each tree
        for tree_id in range(model.tree_count_):
            pred = model.predict(X, ntree_start=tree_id, ntree_end=tree_id+1)
            predictions.append(pred)
        
        tree_predictions = np.array(predictions)
        
        # Calculate confidence metrics
        pred_std = np.std(tree_predictions, axis=0)
        pred_mean = np.mean(tree_predictions, axis=0)
        
        # Calculate relative uncertainty
        relative_uncertainty = pred_std / (pred_mean + 1e-6)
        
        # Transform to 0-1 scale with meaningful spread
        confidence = 1 - relative_uncertainty
        confidence = np.clip(confidence, 0, 1)
        
        return confidence.round(3)

    def make_predictions(self, trained_model, data):
        """Make predictions on data without target column."""
        self.logger.info(f"Making predictions for {self.name}")
        self.logger.info(f"Prediction data columns: {data.columns.tolist()}")
        
        # Make predictions and ensure 1D array
        predictions = trained_model.predict(data).ravel()
        self.logger.info(f"Predictions shape: {predictions.shape}")
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': range(len(predictions)),  # Generate sequence of IDs
            'Premium Amount': predictions.round(3)
        })
        
        # Save predictions
        intermediary_path = self.output_dir / f"{self.name}_predictions.csv"
        self.save_intermediary_submission(submission, intermediary_path)
        
        return None

    def train_model(self, model, data, target):
        self.logger.info(f"Base default train_model for {self.name}")
        X = data
        y = target
        model.fit(X, y)
        return model
    
    def save_intermediary_submission(self, submission, intermediary_path):
        """Save submission DataFrame to CSV, creating output directory if needed."""
        # Create output directory if it doesn't exist 
        intermediary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save submission to CSV
        submission.to_csv(intermediary_path, index=False)

    def save_intermediary(self, model):
        self.logger.info(f"Base default save_intermediary for {self.name}")
        for dataset in ['train', 'validate', 'test']:
            if dataset == 'train': 
                data = data_manager.finaltrain.copy() 
            elif dataset == 'validate':
                data = data_manager.finalvalidate.copy()  
            elif dataset == 'test': 
                data = data_manager.finaltest.copy() 

            X = self.dropIdPremium(data)
            y = data['PremiumAmount']
            predictions = model.predict(X).ravel()
            if dataset != 'test':
                print (f"y: {y.shape} predictions: {predictions.shape}")
                score = safe_rmsle(y, predictions)
                text_score = f"{score:.3f}"
                # Print score
                self.logger.info(f"{dataset} score: {text_score}")
                intermediary_path = self.output_dir / f"{self.name}_{dataset}_estimates_{text_score}.csv"
                # Print path
                self.logger.info(f"{dataset} intermediary results saved to {intermediary_path}")
            else:
                intermediary_path = self.output_dir / f"{self.name}_{dataset}_estimates.csv"
            # Create and save submission
            submission = pd.DataFrame({
                'id': data['id'],
                'Premium Amount': predictions.round(3)
            })
            self.save_intermediary_submission(submission, intermediary_path)
            self.logger.info(f"{dataset} intermediary results saved to {intermediary_path}")

    def save_predictions(self, model, data, split_name):
        """Save predictions and optionally score for a dataset"""
        self.logger.info(f"Processing predictions for {split_name}")
        
        # Prepare features
        X = self.dropIdPremium(data.copy())
        
        # Generate predictions
        predictions = model.predict(X).ravel()
        print (f"predictions: {predictions.shape}", predictions)
        # Calculate score if target exists
        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3)
        })
        
        # Save predictions
        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

    @abstractmethod
    def run(self) -> dict:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the experiment"""
        pass
        
    @abstractmethod
    def create_model(self):
        """Create and configure the model"""
        pass


