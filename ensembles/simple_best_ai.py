import dask.dataframe as dd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from dask.distributed import Client, LocalCluster  # For parallelization with Dask
from pathlib import Path
import os
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiments.data_manager import data_manager

# Initialize a local Dask cluster. Adjust n_workers as needed for your resources
cluster = LocalCluster(n_workers=os.cpu_count(), threads_per_worker=2)  # Customize resources as needed
client = Client(cluster)  # Start Dask Client


# Function to load model data using Dask
def get_model_data(type_of_data):
    model_names = ['catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost']
    models_data = {}
    for model_name in model_names:
        file = f"ensembles/inputs/{model_name}/{type_of_data}.csv"
        if os.path.exists(file):
            models_data[model_name] = dd.read_csv(file) #Use dask for data loading
    return models_data

# Calculate MAE (no change needed, as this function operates on NumPy arrays after NaN removal)
def calculate_mae(actual, predicted):
    actual_clean = np.array(actual)[~np.isnan(predicted)]
    predicted_clean = np.array(predicted)[~np.isnan(predicted)]
    return mean_absolute_error(actual_clean, predicted_clean)



def get_best_model_accuracy(finaltrain, model_predictions, train_ids):
    print('Getting best model accuracy (Dask)...')
    train_ids_set = set(train_ids)

    results = []
    for model in model_predictions:

        # Dask operations:
        predicted_premiums = finaltrain[f'Amt_{model}']
        record_errors = np.abs(finaltrain['PremiumAmount'] - predicted_premiums)

        combined_errors = None  # Initialize combined_errors for the model

        for i in range(1, 21):  # NN1 to NN20
            nn_ids = finaltrain[f'NN{i}']
            nn_premium_col = f'NN{i}_Amt'
            nn_actuals = finaltrain[nn_premium_col]

            # Create a boolean mask for valid neighbors using Dask
            valid_neighbors = nn_ids.isin(train_ids_set)
            
            neighbor_predictions = predicted_premiums.where(valid_neighbors)


            neighbor_errors = (np.abs(nn_actuals-neighbor_predictions)).fillna(0)  # Fill NaN with 0

            if combined_errors is None:
                combined_errors = neighbor_errors
            else:
                combined_errors = combined_errors + neighbor_errors
        # compute per-neighbor error rates
        num_valid_neighbors = finaltrain[[f'NN{i}' for i in range(1, 21)]].isin(train_ids_set).sum(axis=1)

        combined_errors = combined_errors/num_valid_neighbors

        # Convert Dask Series to NumPy for argmin (bottleneck avoidance). Could also use map_partitions to handle on workers.
        record_errors_np = record_errors.compute()
        combined_errors_np = combined_errors.compute()

        best_model_record = np.argmin(record_errors_np)  # Index of best model
        best_model_combined = np.argmin(combined_errors_np) # index of best overall


        results.append({
            'id': finaltrain.iloc[best_model_record]['id'],  # Access ID directly
            'best_model_record': model,
            'best_model_combined': model,
            'record_error': record_errors_np[best_model_record],
            'combined_error': combined_errors_np[best_model_combined],  # Access the correct error
        })


    return pd.DataFrame(results)



def score_data(data_type, data_manager, finaltrain, get_model_data, selected_columns, get_best_model_accuracy):
    # ... (same as before, except using Dask DataFrames where possible and correcting neighbor ID handling)
        data = getattr(data_manager, f'final{data_type}')
        data = dd.from_pandas(data, npartitions=os.cpu_count())

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

# convert dataframes to Dask dataframes
finaltrain = dd.from_pandas(finaltrain, npartitions=os.cpu_count())
train_merged = dd.from_pandas(train_merged, npartitions=os.cpu_count())
result_df = dd.from_pandas(result_df, npartitions=os.cpu_count())

train_ids = finaltrain['id'].values # compute these immediately to avoid issues later


accuracy_results = get_best_model_accuracy(result_df, model_predictions, train_ids)

final_results = dd.merge(result_df, accuracy_results, on='id')

# Compute and print results (trigger computation)
print(final_results.compute().head())
#...
