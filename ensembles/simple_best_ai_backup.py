import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Function to load model data
def get_model_data(type_of_data):
    model_names = ['catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost']
    models_data = {}
    for model_name in model_names:
        file = f"ensembles/inputs/{model_name}/{type_of_data}.csv"
        if os.path.exists(file):
            models_data[model_name] = pd.read_csv(file)
    return models_data

# Load predictions for 'train' data
model_predictions = get_model_data('train')

# Merge predictions into the main DataFrame
finaltrain = data_manager.finaltrain
nn_train = data_manager.nn_train

# Merge the DataFrames
train_merged = pd.merge(finaltrain, nn_train, on='id', how='left')

# Check the columns after merge
original_columns = train_merged.columns

# Define the expected new column names
new_columns = ['id', 'PremiumAmount'] + [f'NN{i}' for i in range(1, 21)] + [f'NN{i}_Amt' for i in range(1, 21)]

# Create a renaming dictionary
renaming_dict = {f'PRED{i}': f'NN{i}_Amt' for i in range(1, 21)}

# Rename the columns
train_merged.rename(columns=renaming_dict, inplace=True)

# Rename columns if the length matches
if len(original_columns) == len(new_columns):
    train_merged.columns = new_columns
else:
    print("Length mismatch: adjusting column names")
    # Adjust this part based on the actual column names and order
    # For example, if there are extra columns, add them to the new_columns list accordingly

# Check if renaming worked as expected
print(train_merged.head())

# Extract the required columns
selected_columns = ['id', 'PremiumAmount'] + [f'NN{i}' for i in range(1, 21)] + [f'NN{i}_Amt' for i in range(1, 21)]

# save a result dataset with only the selected columns to a csv file
# Create a new DataFrame with only the selected columns
result_df = train_merged[selected_columns]
# Add model predictions to the merged DataFrame
for model, predictions in model_predictions.items():
    result_df = result_df.merge(predictions[['id', 'Premium Amount']], on='id', how='left') # Merge with original column name
    #if model == 'catboost':  # Special handling for catboost
    #    result_df.rename(columns={'Premium Amount': 'Amt'}, inplace=True) # Rename only in the result df
    #else:
    result_df.rename(columns={'Premium Amount': f'Amt_{model}'}, inplace=True) # Correctly rename other model predictions

# Save result_df to an output csv file
result_df.to_csv('result_df.csv', index=False)

# Calculate the MAE for each model
def calculate_mae(actual, predicted):
    # Remove NaN values before calculating MAE
    actual_clean = np.array(actual)[~np.isnan(predicted)]
    predicted_clean = np.array(predicted)[~np.isnan(predicted)]
    return mean_absolute_error(actual_clean, predicted_clean)


def get_best_model_accuracy(finaltrain, model_predictions, train_ids):
    print('Getting best model accuracy...')
    num_neighbors = 20
    num_records = len(finaltrain)

    # Convert train_ids to a set for faster lookups
    train_ids_set = set(train_ids)

    model_record_counts = {model: 0 for model in model_predictions}
    model_combined_counts = {model: 0 for model in model_predictions}
    results = []


    for model in model_predictions:
        # ... (Existing code for predictions and combined error calculation)
        model_errors = {}  # Initialize INSIDE the loop, for each model
        combined_errors_dict = {} # initialize a dictionary version for O(1) lookup using the model name as key


        for _, row in finaltrain.iterrows(): # Iterate over each individual record for error calculation

            actual_premium = row['PremiumAmount'] # extract actual premium
            nn_ids = [row[f'NN{i}'] for i in range(1, 21) if row[f'NN{i}'] in train_ids_set]
            predicted_premium = row[f'Amt_{model}']
            predicted_nn_premiums = []
            for nn_id in nn_ids:
                if nn_id in train_ids:
                    predicted_nn_premiums.append(finaltrain[finaltrain['id']==nn_id][f'Amt_{model}'].values[0])
                else:
                    predicted_nn_premiums.append(np.nan) # append NaN as a placeholder if the ID does not exist in the training set

            all_predictions = [predicted_premium] + predicted_nn_premiums # Now always length 21
            all_actuals = [actual_premium] + [row[f'NN{i}_Amt'] for i in range(1,21) if row[f'NN{i}'] in train_ids_set]

            record_error = abs(actual_premium - predicted_premium)
            combined_error = calculate_mae(all_actuals, all_predictions) # error calculation

            # Accumulate errors, grouped by index (id)
            model_errors[row['id']] = record_error  # Store record error for this record
            combined_errors_dict[row['id']] = combined_error


        best_model_record = min(model_errors, key=model_errors.get)  # Now works correctly
        best_model_combined = min(combined_errors_dict, key=combined_errors_dict.get)  # Now works correctly
        
        # Update counts
        model_record_counts[best_model_record] += 1
        model_combined_counts[best_model_combined] += 1

        results.append({
            'id': id,
            'best_model_record': best_model_record,
            'best_model_combined': best_model_combined,
            'record_error': model_errors[best_model_record],
            'combined_error': combined_errors[best_model_combined]
        })

    return pd.DataFrame(results), model_record_counts, model_combined_counts

# Calculate and append the best model accuracy data
train_ids = finaltrain['id'].values
accuracy_results, model_record_counts, model_combined_counts = get_best_model_accuracy(result_df, model_predictions, train_ids)
final_results = pd.merge(result_df, accuracy_results, on='id')

print("Final Results DataFrame:")
print(final_results.head())

# Calculate percentages for each model
total_records = len(finaltrain)
record_percentages = {model: (count / total_records) * 100 for model, count in model_record_counts.items()}
combined_percentages = {model: (count / total_records) * 100 for model, count in model_combined_counts.items()}

# Print the percentages
print("\nModel Performance:")
print("{:<15} {:<20} {:<20}".format("Model", "Best on Record", "Best Overall"))
for model in model_predictions.keys():
    print("{:<15} {:<20} {:<20}".format(model, f"{record_percentages[model]:.2f}%", f"{combined_percentages[model]:.2f}%"))

# Scoring block for validation or test data
def score_data(data_type, data_manager, finaltrain, get_model_data, selected_columns, get_best_model_accuracy):
    """Scores and evaluates model performance on validation or test data.

    Args:
        data_type: 'validate' or 'test'.
        data_manager: Your data manager object with finalvalidate and finaltest.
        finaltrain: The training dataframe.
        get_model_data: Function to load model predictions.
        selected_columns: Columns used for the nearest neighbor data.
        get_best_model_accuracy: Function to calculate best model accuracy.


    Returns: None (prints results and saves submission if data_type is 'test').
    """

    data = getattr(data_manager, f'final{data_type}')  # Access either finalvalidate or finaltest

    # Merge NN data
    merged_data = pd.merge(data, data_manager.nn_train, on='id', how='left')  # merge against NN training data

    # Rename NN prediction columns
    renaming_dict = {f'PRED{i}': f'NN{i}_Amt' for i in range(1, 21)}
    merged_data.rename(columns=renaming_dict, inplace=True)
    
    result_df = merged_data[selected_columns]

    # Merge Model Predictions
    model_predictions = get_model_data(data_type)  # Load predictions for the specific data_type

    for model, predictions in model_predictions.items():
        result_df = result_df.merge(predictions[['id', 'Premium Amount']], on='id', how='left')
        result_df.rename(columns={'Premium Amount': f'Amt_{model}'}, inplace=True)
    

    train_ids = finaltrain['id'].values
    _, model_record_counts, model_combined_counts = get_best_model_accuracy(result_df, model_predictions, train_ids)
    
    record_percentages = {model: (count / len(data)) * 100 for model, count in model_record_counts.items()}
    combined_percentages = {model: (count / len(data)) * 100 for model, count in model_combined_counts.items()}

    print(f"\n{data_type.capitalize()} Data Model Performance:")
    print("{:<15} {:<20} {:<20}".format("Model", "Best on Record", "Best Overall"))
    for model in model_predictions.keys():
        print("{:<15} {:<20} {:<20}".format(model, f"{record_percentages[model]:.2f}%", f"{combined_percentages[model]:.2f}%"))



    if data_type == 'test':
        record_predictions = result_df[['id']].copy()  # Create a copy to avoid SettingWithCopyWarning
        combined_predictions = result_df[['id']].copy()

        for _, row in result_df.iterrows():
            id = row['id']
            best_record_model = row['best_model_record']
            best_combined_model = row['best_model_combined']

            best_record_prediction = row[f'Amt_{best_record_model}']  # Use Amt_ for consistency
            best_combined_prediction = row[f'Amt_{best_combined_model}']

            record_predictions.loc[record_predictions['id'] == id, 'PremiumAmount'] = best_record_prediction
            combined_predictions.loc[combined_predictions['id'] == id, 'PremiumAmount'] = best_combined_prediction

        record_predictions.to_csv('submission_record.csv', index=False)
        combined_predictions.to_csv('submission_combined.csv', index=False)

# To score validation data
score_data('validate', data_manager, finaltrain, get_model_data, selected_columns, get_best_model_accuracy) # Pass all required arguments

# To score test data and generate submission files
# score_data('test')
