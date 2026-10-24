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
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

def start(filename, record, combined):
    # Now we start with two files, the record file and the combined file
    # record file has id, actual_value, closest_model, closest_prediction, closest_model_accuracy
    # id,actual_value,closest_model,closest_prediction,closest_model_accuracy
    # 404339.0,2386.0,randomforest,1949.9,0.8172254819782062
    # combined file has id, actual_value, best_model, best_prediction, best_model_accuracy 
    # id,actual_value,best_model,best_prediction,best_model_accuracy
    # 404339.0,2386.0,randomforest,1949.9,0.8172254819782062
    # we will score three ways, the best model, the best model combined, and the best model combined with the best model
    # we will also output the best model combined with the best model to a csv file
    # we will also output a submission file for the test data, using the best model combined
    # we will also output a submission file for the test data, using the best model
    # we will also output a submission file for the test data, using the best model combined with the best model
    # we will produce a table showing the performance of each model in the best model and the best model combined
    # we will produce a table showing the performance of each model in the best model combined with the best model
    # we will produce a list of outliers that performed poorly in all combinations, to find a remedy
    #
    # Calculate record RMSLE for the entire DataFrame
    train = data_manager.finaltrain.copy()
    train = train[['id', 'PremiumAmount']]
    actual_values = record['actual_value']
    closest_predictions = record['closest_prediction']
    rmsle = np.sqrt(mean_squared_log_error(actual_values, closest_predictions))
    print(f'RMSLE for record: {rmsle}')
    # Calculate combined RMSLE for the entire DataFrame
    combined = pd.merge(train, combined, on='id')
    actual_values = combined['PremiumAmount']
    best_predictions = combined['best_model_prediction']
    rmsle = np.sqrt(mean_squared_log_error(actual_values, best_predictions))
    print(f'RMSLE for combined: {rmsle}')
    # Calculate best of both RMSLE for a bestmerged DataFrame
    bestmerged = pd.merge(combined, record, on='id')
    actual_values = bestmerged['actual_value']
    #take the best prediction from 'best_prediction' and 'closest_prediction'
    bestmerged_predictions = np.where(bestmerged['best_model_accuracy'] > bestmerged['closest_model_accuracy'], bestmerged['best_model_prediction'], bestmerged['closest_prediction'])
    rmsle = np.sqrt(mean_squared_log_error(actual_values, bestmerged_predictions))
    print(f'RMSLE for bestmerged: {rmsle}')
    # now output bestmerged as a submission file
    submission = pd.DataFrame({
        'id': bestmerged['id'],
        'Premium Amount':  np.where(bestmerged['best_model_accuracy'] > bestmerged['closest_model_accuracy'], bestmerged['best_model_prediction'], bestmerged['closest_prediction'])
    })
    submission.to_csv("ensembles/intermediate/{filename}_submission.csv", index=False)

    print("Best Model Performance")
    print(record['closest_model'].value_counts())
    print("Best Model Combined Performance")
    print(combined['best_model'].value_counts())
    # now produce a table showing the performance of each model in the best model combined with the best model
    print("Best Merged Model Combined with Best Model Performance")
    print(bestmerged['closest_model'].value_counts())
    print(bestmerged['best_model'].value_counts())
    # now produce a list of outliers that performed poorly in all combinations, to find a remedy
    print("Outliers")
    print(record[record['closest_model_accuracy'] < 0.5])
    print(combined[combined['best_model_accuracy'] < 0.5])


#start the program
if __name__ == "__main__":
    data_dir = Path("ensembles/intermediate")  # Path to the merged data files
    if len(sys.argv) < 2:
        print("Please provide an argument --train or --validate or --test")
        sys.exit(1)
    if sys.argv[1] == "--train":
        filename = "train"
    elif sys.argv[1] == "--validate":
        filename = "validate"
    elif sys.argv[1] == "--test":
        filename = "test"
    record = pd.read_csv(f"ensembles/intermediate/{filename}_record.csv")
    combined = pd.read_csv(f"ensembles/intermediate/{filename}_combined.csv")
    start(filename, record, combined)
