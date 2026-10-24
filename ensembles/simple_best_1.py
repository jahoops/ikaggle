import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager
from utils import feature_cols

def getModelFileList(model_names, type_of_data):
    models_data = {}
    for model_name in model_names:
        file = f"ensembles/inputs/{model_name}/{type_of_data}.csv"
        if os.path.exists(file):
            models_data[model_name] = pd.read_csv(file)
    return models_data

def getTrainValidateTestSets(data_dir, model_names):
    train = data_manager.finaltrain.copy()
    validate = data_manager.finalvalidate.copy()
    test = data_manager.finaltest.copy()
    
    # Ensure 'id' columns are integers
    train['id'] = train['id'].astype(int)
    validate['id'] = validate['id'].astype(int)
    test['id'] = test['id'].astype(int)
    
    # Merge the train, validate, and test data with the model predictions
    file_paths = [
        {'file': train, 'name': 'train'},
        {'file': validate, 'name': 'validate'},
        {'file': test, 'name': 'test'}
    ]
    for file in file_paths:
        model_files = getModelFileList(model_names, file['name'])
        for model_name in model_files:
            # Rename Premium Amount column to {model_name}
            model_files[model_name] = model_files[model_name][['id', 'Premium Amount']]
            model_files[model_name]['id'] = model_files[model_name]['id'].astype(int)  # Ensure 'id' is integer
            model_files[model_name] = model_files[model_name].rename(columns={'Premium Amount': model_name})
            file['file'] = file['file'].merge(model_files[model_name], on='id')
    
    # Combine all data for scaling
    combined_data = pd.concat([train, validate, test], axis=0)
    
    # Identify features to scale
    features_to_scale = combined_data.columns.difference(['id', 'Premium Amount'])
    
    # Standardize the features
    scaler = RobustScaler()
    combined_data[features_to_scale] = scaler.fit_transform(combined_data[features_to_scale])
    
    # Split the combined data back into train, validate, and test sets
    train = combined_data.iloc[:len(train)]
    validate = combined_data.iloc[len(train):len(train) + len(validate)]
    test = combined_data.iloc[len(train) + len(validate):]
    
    # Ensure feature names are consistent
    train.columns = feature_cols
    validate.columns = feature_cols
    test.columns = feature_cols
    
    # Save these three files to the ensembles/intermediate directory as csv
    train.to_csv(f"{data_dir}/merged_train.csv", index=False)
    validate.to_csv(f"{data_dir}/merged_validate.csv", index=False)
    test.to_csv(f"{data_dir}/merged_test.csv", index=False)

# Start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    model_names = ['catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost']
    getTrainValidateTestSets(data_dir, model_names)
    print("Data sets created")





