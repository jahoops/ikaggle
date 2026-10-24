import pandas as pd
from pathlib import Path

class DataManager:
    """Handles data loading and management for experiments"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent / 'data'
        self.finaltrain = None
        self.finalvalidate = None
        self.finaltest = None

    def load_data(self) -> None:
        """Load train, validation and test datasets"""
        try:
            # Load data from CSV files
            self.finaltrain = pd.read_csv(self.data_dir / 'train.csv')
            self.finalvalidate = pd.read_csv(self.data_dir / 'validate.csv')
            self.finaltest = pd.read_csv(self.data_dir / 'test.csv')
            
            print(f"Loaded datasets:")
            print(f"Train: {len(self.finaltrain)} rows")
            print(f"Validate: {len(self.finalvalidate)} rows")
            print(f"Test: {len(self.finaltest)} rows")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. Please ensure data files exist.") from e

# Global instance
data_manager = DataManager()
