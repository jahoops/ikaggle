import logging
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error
from experiments.base_experiment import BaseExperiment
from experiments import data_manager, safe_rmsle

def safe_rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error in a safe manner."""
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

logger = logging.getLogger(__name__)

class InsuranceNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=50, batch_size=128, learning_rate=0.001):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        X = torch.FloatTensor(X.values).to(self.device)
        y = torch.FloatTensor(y.values).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = InsuranceNet(X.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # Stop after 5 epochs without improvement
                    break
                    
        return self
    
    def predict(self, X):
        self.model.eval()
        X = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu().numpy()

class NeuralNetExperiment(BaseExperiment):
    @property
    def name(self) -> str:
        return "NeuralNet"
        
    def create_model(self):
        return PyTorchRegressor()
    
    def get_param_distributions(self):
        if self.test_mode:
            return {
                'epochs': [1], # , 50],
                'batch_size': [32], # , 128],
                'learning_rate': [0.001]
            }
        else:
            return {
                'epochs': [10, 20, 50],
                'batch_size': [32, 64, 128],
                'learning_rate': [0.001, 0.0001]
            }
            
    def run(self):
        start_time = time.time()
        self.logger.info(f"[{self.name}] Starting run at {self.format_time(start_time)}")
        
        # Load training data
        train_data = data_manager.finaltrain.copy()
        X_train = self.dropIdPremium(train_data)
        y_train = train_data['PremiumAmount']

        # Perform hyperparameter search on PyTorchRegressor
        param_distributions = self.get_param_distributions()
        self.logger.info(f"Param distributions: {param_distributions}")
        
        search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=1 if self.test_mode else 10,
            cv=RepeatedKFold(n_splits=3, n_repeats=1, random_state=42),
            scoring='neg_mean_squared_error',
            verbose=1
        )
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        self.logger.info(f"Best params: {search.best_params_}")

        # Save predictions for train/validate/test
        self.add_predictions(best_model, data_manager.finaltrain, 'finaltrain')
        self.add_predictions(best_model, data_manager.finalvalidate, 'finalvalidate')
        self.add_predictions(best_model, data_manager.finaltest, 'finaltest')

        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
        return best_model

    def add_predictions(self, model, data, split_name):
        X = self.dropIdPremium(data)
        raw_preds = model.predict(X)  # might have shape (N, 1)
        predictions = raw_preds.ravel()

        # Placeholder for confidence: zeros
        confidence = np.zeros_like(predictions)

        if split_name in ['finaltrain', 'finalvalidate']:
            score = safe_rmsle(data['PremiumAmount'], predictions)
            filename = f"{self.name}_{split_name}_estimates_{score:.3f}.csv"
            self.logger.info(f"{split_name} score: {score:.3f}")
        else:
            filename = f"{self.name}_{split_name}_estimates.csv"

        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3),
            'Confidence': confidence.round(3)
        })

        output_path = self.output_dir / filename
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {output_path}")

    def create_model(self):
        """Create a new instance of PyTorchRegressor"""
        return PyTorchRegressor()

    def get_param_distributions(self):
        """Return parameter distributions for RandomizedSearchCV"""
        if self.test_mode:
            return {
                'epochs': [10],
                'batch_size': [32],
                'learning_rate': [0.001]
            }
        else:
            return {
                'epochs': [25, 50],
                'batch_size': [64, 128],
                'learning_rate': [0.001, 0.0005]
            }

def run_neuralnet(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = NeuralNetExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    run_neuralnet('output', test_mode=True)