import pandas as pd
import numpy as np

# Load your dataset
data = pd.DataFrame(columns=all_columns)  # Replace with your actual data loading code

# Calculate correlations
correlation_matrix = data.corr()

# Display correlations with nearest neighbor features
nn_features = [col for col in data.columns if 'NN' in col or 'DIST' in col]
correlations_with_nn = correlation_matrix.loc[nn_features, :]

print(correlations_with_nn)
