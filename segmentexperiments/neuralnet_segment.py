from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .base_segment_experiment import BaseSegmentExperiment
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.amp import autocast  # Change this line
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Add MinMaxScaler import

class InsuranceDataset(Dataset):
    def __init__(self, features: pd.DataFrame, targets: np.ndarray = None):
        """
        Initialize Insurance dataset
        Args:
            features: DataFrame containing feature columns
            targets: numpy array of target values (optional)
        """
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class InsuranceNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

class RMSLELoss(nn.Module):
    def forward(self, y_pred, y_true):
        eps = 1e-6
        
        # Add detailed debugging prints
        # print(f"\nTarget stats - min: {y_true.min().item():.4f}, max: {y_true.max().item():.4f}")
        # print(f"Pred stats - min: {y_pred.min().item():.4f}, max: {y_pred.max().item():.4f}")
        
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print(f"NaN detected - y_pred: {torch.isnan(y_pred).sum()}, y_true: {torch.isnan(y_true).sum()}")
        
        if (y_pred < 0).any() or (y_true < 0).any():
            print(f"Negative values detected - y_pred: {(y_pred < 0).sum()}, y_true: {(y_true < 0).sum()}")
        
        # Ensure positive values
        y_pred = torch.clamp(y_pred, min=eps)
        y_true = torch.clamp(y_true, min=eps)
        
        return torch.sqrt(torch.mean((torch.log1p(y_pred) - torch.log1p(y_true))**2))

class NeuralNetSegmentExperiment(BaseSegmentExperiment):
    """Neural network-based segment experiment with GPU support"""
    
    def __init__(self, output_dir: Path, test_mode: bool = False):
        # Initialize features first
        self.numeric_features = [
            'Age',
            'AnnualIncome', 
            'HealthScore',
            'CreditScore',
            'NumberofDependents'
        ]
        self.feature_columns = self.numeric_features.copy()
        self.feature_order = self.feature_columns.copy()
        self.model = None
        
        # Initialize base class
        super().__init__(output_dir, test_mode)
        
        # Verify CUDA availability before proceeding
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This experiment requires GPU acceleration.\n"
                "Please check:\n"
                "1. NVIDIA drivers are installed\n"
                "2. PyTorch is installed with CUDA support\n"
                "3. A compatible GPU is present\n"
                "\nTo install PyTorch with CUDA support:\n"
                "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
            )
        
        # Initialize CUDA device and scaler
        self.device = torch.device('cuda')
        self.grad_scaler = GradScaler()
        
        # Test CUDA functionality
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CUDA: {e}")
        
        self.scaler = StandardScaler()  # Keep StandardScaler for features
        self.target_scaler = MinMaxScaler()  # Change to MinMaxScaler for target

    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            if not torch.cuda.is_available():
                print("CUDA not available")
                return False
            
            # Test CUDA initialization
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            return True
        except Exception as e:
            print(f"GPU error: {e}")
            return False

    def create_model(self) -> BaseEstimator:
        """Create neural network model with optimized architecture"""
        class OptimizedInsuranceNet(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.network = nn.Sequential(
                    # Input layer with batch normalization
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    
                    # Hidden layers with residual connections
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    # Output layer
                    nn.Linear(128, 1),
                    nn.ReLU()  # Ensure non-negative predictions
                )
            
            def forward(self, x):
                return self.network(x).squeeze()

        return OptimizedInsuranceNet(len(self.feature_columns)).to(self.device)

    def get_quick_test_params(self) -> Dict[str, Any]:
        """Return parameters for quick testing"""
        return {
            'batch_size': 1024,
            'epochs': 10,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5
        }

    def get_model_params(self) -> Dict[str, Any]:
        """Return optimized parameters for full training"""
        return {
            'batch_size': 512,        # Smaller batches for better generalization
            'epochs': 200,            # More epochs with early stopping
            'learning_rate': 3e-4,    # Standard for Adam optimizer
            'weight_decay': 1e-4      # Increased regularization
        }

    def save_model(self, epoch: int, best_loss: float) -> Path:
        """Save model checkpoint with metadata
        
        Args:
            epoch: Current training epoch
            best_loss: Best loss achieved so far
        
        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_loss': best_loss,
            'feature_columns': self.feature_columns,
            'device': str(self.device)
        }
        
        # Save to models directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.models_dir / f'model_checkpoint_{timestamp}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path

    def load_model(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load model checkpoint and metadata
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dictionary with loaded metadata
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        
        # Verify feature columns match
        if checkpoint['feature_columns'] != self.feature_columns:
            raise ValueError("Checkpoint features don't match current model")
        
        # Create and load model
        self.model = InsuranceNet(len(self.feature_columns)).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'best_loss': checkpoint['best_loss']
        }

    def train(self, train_data: pd.DataFrame) -> None:
        """Train neural network with validation and checkpointing"""
        # Verify CUDA status at start of training
        if self.device.type == 'cuda':
            print(f"Training on GPU: {torch.cuda.get_device_name()}")
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        else:
            print("Warning: Training on CPU - CUDA not available")
        
        params = self.get_quick_test_params() if self.test_mode else self.get_model_params()
        
        # Scale features before creating dataset
        scaled_features = self.scaler.fit_transform(train_data[self.feature_columns])
        
        # Scale target separately
        scaled_targets = self.target_scaler.fit_transform(
            train_data['PremiumAmount'].values.reshape(-1, 1)
        ).ravel()
        
        scaled_train_data = train_data.copy()
        scaled_train_data[self.feature_columns] = scaled_features
        
        dataset = InsuranceDataset(
            scaled_train_data[self.feature_columns],
            scaled_targets  # Use scaled targets
        )
        # Print 5 train_data samples
        print(train_data.sample(5)) 
        
        dataloader = DataLoader(
            dataset, 
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0  # No multiprocessing for now
        )
        
        # Validate scaling
        print("\nScaling Statistics:")
        print(f"Original target range: [{train_data['PremiumAmount'].min()}, {train_data['PremiumAmount'].max()}]")
        print(f"Scaled target range: [{scaled_targets.min()}, {scaled_targets.max()}]")
        
        # Check for NaN values after scaling
        if np.isnan(scaled_features).any():
            raise ValueError("NaN values found in scaled features")
        if np.isnan(scaled_targets).any():
            raise ValueError("NaN values found in scaled targets")
        
        # Initialize model
        self.model = InsuranceNet(len(self.feature_columns)).to(self.device)
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        criterion = RMSLELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(params['epochs']):
            epoch_loss = 0.0
            for features, targets in dataloader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Use mixed precision training
                with autocast('cuda'):  # Change this line
                    outputs = self.model(features)
                    loss = criterion(outputs, targets)
                
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()
                
                epoch_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{params['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

        original_train_data = self.revert_scaled_features(train_data)
        # Now original_train_data[self.numeric_features] are back to unscaled values

    def validate(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Validate model and return metrics"""
        if self.model is None:
            raise RuntimeError("Model must be trained before validation")
            
        # Keep original IDs
        ids = val_data['id'].copy() if 'id' in val_data.columns else None
        
        # Scale validation features using the same scaler
        scaled_features = self.scaler.transform(val_data[self.feature_columns])
        X_val = torch.FloatTensor(scaled_features).to(self.device)
        y_val = val_data['PremiumAmount'].values
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            scaled_pred = self.model(X_val)
            # Print intermediate values
            print(f"\nValidation Statistics:")
            print(f"Scaled predictions range: [{scaled_pred.min().item()}, {scaled_pred.max().item()}]")
            
            # Reshape and inverse transform
            y_pred = self.target_scaler.inverse_transform(scaled_pred.cpu().numpy().reshape(-1, 1)).ravel()
            print(f"Unscaled predictions range: [{y_pred.min()}, {y_pred.max()}]")
        
        # Use base class metrics calculation
        metrics = self.calculate_metrics(y_val, y_pred, 'validation')
        
        # Save predictions if we have IDs
        if ids is not None:
            self.save_predictions_with_ids(ids, y_pred, 'validation')
            
        return metrics

    def test(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """Generate predictions for test data"""
        if self.model is None:
            raise RuntimeError("Model must be trained before testing")
            
        # Keep original IDs
        ids = test_data['id'].copy() if 'id' in test_data.columns else None
        if ids is None:
            raise ValueError("Test data must include 'id' column")
        
        # Prepare test data
        scaled_features = self.scaler.transform(test_data[self.feature_columns])
        X_test = torch.FloatTensor(scaled_features).to(self.device)
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            scaled_pred = self.model(X_test).cpu().numpy().ravel()
            # Inverse transform predictions
            y_pred = self.target_scaler.inverse_transform(
                scaled_pred.reshape(-1, 1)
            ).ravel()
        
        # Save predictions with IDs
        self.save_predictions_with_ids(ids, y_pred, 'test')
        
        # Return basic metrics about predictions
        return {
            'test_samples': len(y_pred),
            'mean_prediction': float(np.mean(y_pred)),
            'min_prediction': float(np.min(y_pred)),
            'max_prediction': float(np.max(y_pred))
        }

    def get_experiment_metadata(self) -> Dict:
        """Return metadata for the experiment including GPU metrics and training scores"""
        metadata = {
            "name": "NeuralNetSegmentExperiment",
            "test_mode": self.test_mode,
            "timestamp": datetime.now().isoformat(),
            "hardware_config": {
                "gpu_enabled": torch.cuda.is_available(),
                "device_name": None,
                "memory_allocated": None,
                "compute_capability": None
            }
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device)
                
                # Get memory information
                memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)    # MB
                
                metadata["hardware_config"].update({
                    "device_name": props.name,
                    "memory_allocated": float(memory_allocated),
                    "memory_reserved": float(memory_reserved),
                    "total_memory": float(props.total_memory / (1024**2)),
                    "compute_capability": f"{props.major}.{props.minor}"
                })
            except Exception as e:
                print(f"Warning: Could not get detailed GPU properties: {e}")
        
        # Add model information
        if hasattr(self, 'model') and self.model is not None:
            # Count model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            metadata.update({
                "model_info": {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "architecture": str(self.model),
                    "input_features": len(self.feature_columns)
                }
            })
            
            # Add training configuration if available
            if hasattr(self, 'best_loss'):
                metadata["training_info"] = {
                    "best_loss": float(self.best_loss),
                    "epochs_trained": self.current_epoch if hasattr(self, 'current_epoch') else 0,
                    "learning_rate": self.get_quick_test_params()['learning_rate'] if self.test_mode else self.get_model_params()['learning_rate']
                }
        
        return metadata

    def save_predictions_with_ids(self, ids: pd.Series, predictions: np.ndarray, dataset_type: str) -> None:
        """Save predictions with IDs to standardized location"""
        # Create predictions directory
        preds_dir = self.output_dir / 'predictions'
        preds_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standardized filename
        output_path = preds_dir / f'{dataset_type}_predictions.csv'
        print(f"Saving {dataset_type} predictions to: {output_path}")
        
        # Save predictions
        pd.DataFrame({
            'id': ids,
            'PremiumAmount': predictions
        }).to_csv(output_path, index=False)

    def revert_scaled_features(self, scaled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Revert the scaled numeric features to their original values.
        """
        original_df = scaled_df.copy()
        original_df[self.numeric_features] = self.scaler.inverse_transform(scaled_df[self.numeric_features])
        return original_df