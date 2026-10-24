import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, RegressorMixin
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from experiments import data_manager

logger = logging.getLogger(__name__)

class MetaNet(nn.Module):
    def __init__(self, input_dim, test_mode):
        super(MetaNet, self).__init__()
        hidden_dim = 8 if test_mode else 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MetaLearner:
    def __init__(self, input_dir='ensembles/inputs', output_dir='ensembles/outputs', test_mode=False):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.test_mode = test_mode
        
        # Make sure to define self.device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Adjust parameters
        print ("Test mode: ", test_mode)
        self.epochs = 10 if test_mode else 100
        self.batch_size = 32 if test_mode else 128
        self.learning_rate = 0.01 if test_mode else 0.0005  # More conservative LR for longer training
        self.patience = 3 if test_mode else 10
        
        # Track losses
        self.train_losses = []
        self.val_losses = []
        
        self.model = None

    def train(self):
        train_data = data_manager.finaltrain.copy()
        validate_data = data_manager.finalvalidate.copy()
        test_data = data_manager.finaltest.copy()
        
        # Use correct column names
        feature_cols = [col for col in train_data.columns if col not in ['id', 'PremiumAmount']]
        X_train = train_data[feature_cols]
        y_train = train_data['PremiumAmount']
        
        self.fit(X_train, y_train)
        
        train_preds = self.predict(X_train)
        validate_preds = self.predict(validate_data[feature_cols])
        test_preds = self.predict(test_data[feature_cols])
        
        # Calculate metrics
        train_rmsle = safe_rmsle(y_train.values, train_preds)
        validate_rmsle = safe_rmsle(validate_data['PremiumAmount'].values, validate_preds)
        
        print(f"Train RMSLE: {train_rmsle:.4f}")
        print(f"Validate RMSLE: {validate_rmsle:.4f}")
        
        self.save_predictions(train_preds, train_data['id'], 'train')
        self.save_predictions(validate_preds, validate_data['id'], 'validate')
        self.save_predictions(test_preds, test_data['id'], 'test')

    def fit(self, X, y):
        X = torch.FloatTensor(X.to_numpy()).to(self.device)
        y = torch.FloatTensor(y.values).reshape(-1, 1).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = MetaNet(X.shape[1], self.test_mode).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader)
            self.train_losses.append(avg_train_loss)
            
            # If you want to validate each epoch, load and compute similarly.
            val_loss = self._validate()  
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.output_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info("Early stopping triggered")
                    break

        self._plot_training_history()

    def _validate(self):
        # Minimal placeholder validation
        # You could use 'validate_data' similarly to train to measure a proper loss
        return 0.0

    def _plot_training_history(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.legend()
        plt.savefig(self.output_dir / 'training_history.png')
        plt.close()

    def predict(self, X):
        self.model.eval()
        X = torch.FloatTensor(X.values).to(self.device)
        with torch.no_grad():
            predictions = self.model(X)
        return predictions.cpu().numpy().flatten()

    def save_predictions(self, preds, ids, split):
        submission = pd.DataFrame({
            'id': ids,
            'Premium Amount': preds.round(3)
        })
        submission.to_csv(self.output_dir / f'meta_{split}_submission.csv', index=False)

def safe_rmsle(y_true, y_pred):
    """Calculate RMSLE safely."""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

if __name__ == '__main__':
    import argparse
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    meta = MetaLearner(test_mode=args.test)
    meta.train()