from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def load_segment_data():
    """Load and prepare data for segment experiments"""
    # Define data paths
    data_dir = Path('/home/USER/Projects/ikaggle/data')
    train_path = data_dir / 'train.csv'
    test_path = data_dir / 'test.csv'
    
    # Load data
    train_full = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Split train into train and validation
    train, validate = train_test_split(
        train_full, 
        test_size=0.2, 
        random_state=42,
        stratify=train_full['NumberofDependents']
    )
    
    return train, validate, test