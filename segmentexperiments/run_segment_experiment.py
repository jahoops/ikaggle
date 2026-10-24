from pathlib import Path
from segmentexperiments.rapids_segment import RapidsSegmentExperiment
from experiments import data_manager
import pandas as pd

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
    from sklearn.model_selection import train_test_split
    train, validate = train_test_split(
        train_full, 
        test_size=0.2, 
        random_state=42,
        stratify=train_full['NumberofDependents']
    )
    
    # Set data manager attributes
    data_manager.finaltrain = train
    data_manager.finalvalidate = validate
    data_manager.finaltest = test
    
    return train, validate, test

def main():
    # Load data
    train, validate, test = load_segment_data()
    
    # Print data shapes for verification
    print("Data shapes:")
    print(f"Train: {train.shape}")
    print(f"Validation: {validate.shape}")
    print(f"Test: {test.shape}")
    
    # Create and run experiment
    experiment = RapidsSegmentExperiment(
        output_dir=Path('/home/USER/Projects/ikaggle/segment_output'),
        test_mode=True
    )
    
    experiment.run(
        train_data=train,
        val_data=validate,
        test_data=test
    )

if __name__ == "__main__":
    main()