# input optionally /ensemble/intermediate/merged_train.csv or /ensemble/intermediate/merged_validate.csv or /ensemble/intermediate/merged_test.csv
# by --train or --validate or --test
# for train and validate, the program will for every record output id, the best model, the estimate, and it's accuracy, best model combined, the estimate, and it's accuracy
# a final print output table will show all models and how they contribute by percentage to the best model on record and the best model combined
# for test, the program will output two csv files, submission_record.csv and submission_combined.csv

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

def start(file):
    returnedFile = find_closest_combined(file)
    returnedFile.to_csv(f"ensembles/intermediate/test_combined.csv", index=False)

def find_closest_model(file):
    # Extract relevant columns
    actual_col = 'PremiumAmount'
    model_cols = ['Amt_catboost', 'Amt_knn', 'Amt_lasso', 'Amt_lightgbm', 'Amt_linear', 'Amt_neuralnet', 'Amt_randomforest', 'Amt_ridge', 'Amt_svr', 'Amt_xgboost']
    
    # Initialize list to store results
    results = []
    
    # Process each record
    for index, row in file.iterrows():
        print(f"Processing row {index} of {len(file)}")
        predictions = row[model_cols]
        
        # Find the model with the closest prediction to the actual value
        abs_diff = abs(predictions - actual_value)
        closest_model = abs_diff.idxmin()
        closest_prediction = predictions[closest_model]
        
        # Calculate accuracy
        closest_model_accuracy = 1 - (abs(closest_prediction - actual_value) / actual_value)
        
        # Store the result
        results.append({
            'id': row['id'],
            'actual_value': row['PremiumAmount'],
            # remove Amt_ from the model name
            'closest_model': closest_model[4:],
            'closest_prediction': closest_prediction,
            'closest_model_accuracy': closest_model_accuracy
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def find_closest_combined(file):
    # Initialize list to store results
    results = []
    # Nearest neighbors are ALWAYS from train data, not validate or test
    nnlookupfile = pd.read_csv(f"ensembles/intermediate/merged_train.csv")
    # Define columns
    model_cols = ['Amt_catboost', 'Amt_knn', 'Amt_lasso', 'Amt_lightgbm', 'Amt_linear', 'Amt_neuralnet', 'Amt_randomforest', 'Amt_ridge', 'Amt_svr', 'Amt_xgboost']
    nn_cols = [f'NN{i}' for i in range(1, 21)]
    
    # Initialize list to store results
    results = []
    
    # Process each record
    for index, row in file.iterrows():
        print(f"Processing row {index} of {len(file)}")
        avg_pred = row['avg_pred']
        median_prod = row['median_prod']
        nn_ids = row[nn_cols]
        
        # Lookup the nn_ids in the nnlookupfile
        nn_data = nnlookupfile[nnlookupfile['id'].isin(nn_ids)]
        
        # Initialize dictionary to store accuracies for each model
        model_accuracies = {}
        
        for model in model_cols:
            # Calculate accuracy for the original record
            model_prediction = row[model]
            
            # Calculate accuracies for nearest neighbors
            nn_model_predictions = nn_data[model]
            nn_avg_pred = nn_data[avg_pred]
            nn_median_pred = nn_data[avg_pred]
            nn_accuracies = 1 - (abs(nn_model_predictions - nn_actual_values) / nn_actual_values)
            
            # Calculate mean accuracy
            mean_nn_accuracy = nn_accuracies.mean()
            
            # Store accuracies in the dictionary
            model_accuracies[model] = {
                'original_prediction': model_prediction,
                
                'mean_nn_accuracy': mean_nn_accuracy,
                'mean_nn_prediction': nn_model_predictions.mean()
            }
        
        # Determine the best model based on mean NN accuracy
        best_model = max(model_accuracies, key=lambda x: model_accuracies[x]['mean_nn_accuracy'])
        best_model_accuracy = model_accuracies[best_model]['mean_nn_accuracy']
        
        # Store the result
        results.append({
            'id': row['id'],
            'best_model': best_model,
            'best_model_accuracy': best_model_accuracy,
            'best_model_prediction': model_accuracies[best_model]['mean_nn_prediction'],
            'model_accuracies': model_accuracies
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

#start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    train = pd.read_csv(f"ensembles/intermediate/merged_test.csv")
    file = pd.read_csv(f"ensembles/intermediate/merged_test.csv")
    start(train, test)
