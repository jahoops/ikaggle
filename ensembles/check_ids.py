import pandas as pd
import os

def check_ids(model_names, type_of_data):
    non_integer_ids = {}
    for model_name in model_names:
        file = f"ensembles/inputs/{model_name}/{type_of_data}.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            if not df['id'].apply(lambda x: isinstance(x, int)).all():
                non_integer_ids[model_name] = df[~df['id'].apply(lambda x: isinstance(x, int))]
    return non_integer_ids

def main():
    model_names = ['catboost', 'knn', 'lasso', 'lightgbm', 'linear', 'neuralnet', 'randomforest', 'ridge', 'svr', 'xgboost']
    data_types = ['train', 'validate', 'test']
    
    for data_type in data_types:
        non_integer_ids = check_ids(model_names, data_type)
        if non_integer_ids:
            print(f"Non-integer IDs found in {data_type} data:")
            for model_name, df in non_integer_ids.items():
                print(f"Model: {model_name}")
                print(df[['id']])
        else:
            print(f"No non-integer IDs found in {data_type} data.")

if __name__ == "__main__":
    main()
