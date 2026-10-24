import pandas as pd
import numpy as np
import os

def find_outliers(df, threshold=1.5):
    """Identify outliers in the dataframe using the IQR method."""
    outliers = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        if set(df[column].unique()) == {0, 1}:  # Skip binary columns
            continue
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def main():
    data_dir = "data/processed"
    files = ['cleaned_train.csv', 'cleaned_validate.csv', 'cleaned_test.csv']
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            outliers = find_outliers(df)
            print(f"Outliers in {file}:")
            outlier_counts = {}
            for column, outlier_df in outliers.items():
                if not outlier_df.empty:
                    outlier_counts[column] = len(outlier_df)
                    print(f"Column: {column}")
                    print(outlier_df[['id', column]])
            # Calculate and display percentage of outliers
            total_rows = len(df)
            outlier_percentages = {column: (count / total_rows) * 100 for column, count in outlier_counts.items()}
            outlier_table = pd.DataFrame.from_dict(outlier_percentages, orient='index', columns=['Outlier Percentage'])
            print("\nOutlier Percentages:")
            print(outlier_table)
        else:
            print(f"File {file_path} not found.")

if __name__ == "__main__":
    main()
