import pandas as pd
from pathlib import Path

def merge_predictions(intermediate_dir, model_dirs):
    # Load the merged train and validate data
    merged_train_path = intermediate_dir / "merged_train.csv"
    merged_validate_path = intermediate_dir / "merged_validate.csv"
    
    merged_train = pd.read_csv(merged_train_path)
    merged_validate = pd.read_csv(merged_validate_path)
    
    for model_dir in model_dirs:
        model_train_path = model_dir / "train.csv"
        model_validate_path = model_dir / "validate.csv"
        
        # Load the model's train and validate predictions
        model_train = pd.read_csv(model_train_path)
        model_validate = pd.read_csv(model_validate_path)
        
        # Merge the predictions with the merged data
        merged_train = merged_train.merge(model_train, on='id', suffixes=('', f'_{model_dir.name}'))
        merged_validate = merged_validate.merge(model_validate, on='id', suffixes=('', f'_{model_dir.name}'))
    
    # Save the updated merged data
    merged_train.to_csv(merged_train_path, index=False)
    merged_validate.to_csv(merged_validate_path, index=False)
    print(f"Updated merged_train.csv and merged_validate.csv saved to {intermediate_dir}")

if __name__ == "__main__":
    intermediate_dir = Path("ensembles/intermediate")
    model_dirs = [Path(f"ensembles/model_{i}") for i in range(1, 11)]  # Assuming model folders are named model_1, model_2, ..., model_10
    merge_predictions(intermediate_dir, model_dirs)
