import logging
import time
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error
from experiments.base_experiment import BaseExperiment
from experiments import data_manager

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
                'epochs': [10], # , 50],
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
        
        data = data_manager.finaltrain.copy()
        X = self.dropIdPremium(data)
        y = data['PremiumAmount']
        
        cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=42)
        param_distributions = self.get_param_distributions()
        
        search = RandomizedSearchCV(
            estimator=self.create_model(),
            param_distributions=param_distributions,
            n_iter=30,
            scoring='neg_mean_squared_error',
            cv=cv,
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        
        search.fit(X, y)
        trained_model = search.best_estimator_
        
        self.logger.info(f"Best params: {search.best_params_}")
        
        # Save predictions using base class method
        self.save_predictions(trained_model, data_manager.finaltrain, 'finaltrain')
        self.save_predictions(trained_model, data_manager.finalvalidate, 'finalvalidate')
        self.save_predictions(trained_model, data_manager.finaltest, 'finaltest')
        
        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"[{self.name}] Finished run at {self.format_time(end_time)}, elapsed {elapsed:.2f}s")
        return trained_model

    def save_predictions(self, model, data, split_name):
        self.logger.info(f"Processing predictions for {split_name}")
        X = self.dropIdPremium(data)
        predictions = model.predict(X).ravel()
        
        # Use RMSE instead of MSLE
        rmse = np.sqrt(mean_squared_error(data["PremiumAmount"], predictions))
        self.logger.info(f"{split_name} RMSE: {rmse:.3f}")
        
        submission = pd.DataFrame({
            'id': data['id'],
            'Premium Amount': predictions.round(3)
        })
        filename = f"{self.name}_{split_name}_estimates_{rmse}.csv"
        out_path = self.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_path, index=False)
        self.logger.info(f"Saved {split_name} predictions to {out_path}")

def run_neuralnet(output_dir, test_mode=False):
    output_path = Path(output_dir)
    experiment = NeuralNetExperiment(output_path, test_mode=test_mode)
    model_info = experiment.run()
    return model_info

if __name__ == "__main__":
    run_neuralnet('output', test_mode=True)