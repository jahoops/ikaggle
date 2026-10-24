import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class SubmissionCNN(nn.Module):
    def __init__(self, n_features, n_submissions):
        super().__init__()
        # Calculate grid size to fit all features
        self.grid_size = max(4, int(np.ceil(np.sqrt(n_features))))
        self.n_features = n_features
        self.n_submissions = n_submissions
        
        # Simpler architecture with correct size calculations
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, padding=1)
        
        # Calculate actual output size after convolution
        conv_output_h = self.grid_size + 2 * 1 - 2 + 1  # height after conv1
        conv_output_w = self.grid_size + 2 * 1 - 2 + 1  # width after conv1
        self.conv_output_size = 16 * conv_output_h * conv_output_w
        
        print(f"Debug - Sizes: grid={self.grid_size}, conv_out={self.conv_output_size}")
        
        self.fc1 = nn.Linear(self.conv_output_size, 256)
        self.fc2 = nn.Linear(256, n_submissions)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Pad and reshape
        padded_size = self.grid_size * self.grid_size
        if x.size(1) < padded_size:
            padding = torch.zeros(batch_size, padded_size - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        x = x.view(batch_size, 1, self.grid_size, self.grid_size)
        
        # Single conv layer
        x = F.relu(self.conv1(x))
        
        # Flatten and fully connected layers
        x = x.view(batch_size, -1)
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class SubmissionSelector:
    def __init__(self, n_submissions, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.n_submissions = n_submissions
        self.model = None
        
    def prepare_data(self, bitbins_data):
        """Prepare binary features and calculate best submission per sample"""
        # Get only feature-based binary columns, explicitly excluding scores and target
        binary_cols = [col for col in bitbins_data.columns 
                      if ('_bin_' in col 
                          and not col.startswith('score_') 
                          and col != 'PremiumAmount')]
        
        print("Features being used:")
        print(binary_cols[:5], "... and", len(binary_cols)-5, "more")
        
        score_cols = [f'score_sub_{i}' for i in range(1, self.n_submissions + 1)]
        print("\nScore columns being used for labels:")
        print(score_cols)
        
        # Convert to tensors
        X = torch.tensor(bitbins_data[binary_cols].values, dtype=torch.float32)
        scores = bitbins_data[score_cols].values
        
        # For each sample, find submission with lowest score
        y = torch.tensor(np.argmin(scores, axis=1), dtype=torch.long)
        
        print(f"\nInput shape: {X.shape}")
        print(f"Unique target values: {torch.unique(y).tolist()}")
        
        return X, y
        
    def train(self, bitbins_data, epochs=50, batch_size=64):
        X, y = self.prepare_data(bitbins_data)
        
        # Create model
        self.model = SubmissionCNN(X.shape[1], self.n_submissions).to(self.device)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val.to(self.device))
                val_loss = criterion(val_outputs, y_val.to(self.device))
                accuracy = (val_outputs.argmax(1) == y_val.to(self.device)).float().mean()
                
            # if (epoch + 1) % 5 == 0:
                # print(f'Epoch [{epoch+1}/{epochs}], Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    def predict(self, bitbins_data):
        """Predict best submission for each sample"""
        if self.model is None:
            raise RuntimeError("Model must be trained first")
            
        binary_cols = [col for col in bitbins_data.columns 
                      if any(col.startswith(prefix) 
                            for prefix in ['Age_bin_', 'Income_bin_', 'Health_bin_'])]
        
        X = torch.tensor(bitbins_data[binary_cols].values, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X.to(self.device))
            predictions = outputs.argmax(1).cpu().numpy()
            
        return predictions + 1  # Convert to 1-based submission IDs

def main():
    # Load scored bitbins
    bitbins_path = Path('/home/USER/Projects/ikaggle/AAA/bitbins/train_bitbins_scored.csv')
    bitbins_data = pd.read_csv(bitbins_path)
    
    # Count number of submissions
    n_submissions = len([col for col in bitbins_data.columns if col.startswith('score_sub_')])
    
    # Create and train selector
    selector = SubmissionSelector(n_submissions)
    selector.train(bitbins_data)
    
    # Create models directory if it doesn't exist
    model_dir = Path('/home/USER/Projects/ikaggle/AAA/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save trained model
    model_path = model_dir / 'submission_cnn.pth'
    torch.save(selector.model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Make predictions
    predictions = selector.predict(bitbins_data)
    
    # Calculate average score using selected submissions
    total_score = 0
    for idx, pred in enumerate(predictions):
        score = bitbins_data.iloc[idx][f'score_sub_{pred}']
        total_score += score
    
    avg_score = total_score / len(predictions)
    print(f"\nAverage RMSLE using CNN selector: {avg_score:.6f}")

if __name__ == "__main__":
    main()
