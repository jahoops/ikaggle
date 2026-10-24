import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from utils import all_columns, data_manager  # Ensure correct import
import logging

logger = logging.getLogger(__name__)

# Load data
train = data_manager.featuretrain.copy()
validate = data_manager.featurevalidate.copy()
test = data_manager.featuretest.copy()

nn_train = data_manager.nn_train.copy()
nn_validate = data_manager.nn_validate.copy()
nn_test = data_manager.nn_test.copy()

# Merge on "id"
train_merged = pd.merge(train, nn_train, on='id', how='left')
validate_merged = pd.merge(validate, nn_validate, on='id', how='left')
test_merged = pd.merge(test, nn_test, on='id', how='left')

# Calculate distance features
dist_cols = [f'DIST{i}' for i in range(1, 21)]
pred_cols = [f'PRED{i}' for i in range(1, 21)]

# Add aggregate features
train_merged['avg_dist'] = train_merged[dist_cols].mean(axis=1)
train_merged['median_dist'] = train_merged[dist_cols].median(axis=1)
train_merged['avg_pred'] = train_merged[pred_cols].mean(axis=1)
train_merged['median_pred'] = train_merged[pred_cols].median(axis=1)
# Add aggregate features
validate_merged['avg_dist'] = validate_merged[dist_cols].mean(axis=1)
validate_merged['median_dist'] = validate_merged[dist_cols].median(axis=1)
validate_merged['avg_pred'] = validate_merged[pred_cols].mean(axis=1)
validate_merged['median_pred'] = validate_merged[pred_cols].median(axis=1)

test_merged['avg_dist'] = test_merged[dist_cols].mean(axis=1)
test_merged['median_dist'] = test_merged[dist_cols].median(axis=1)
test_merged['avg_pred'] = test_merged[pred_cols].mean(axis=1)
test_merged['median_pred'] = test_merged[pred_cols].median(axis=1)

# Reindex columns to match all_columns
final_train = train_merged.reindex(columns=all_columns)
final_validate = validate_merged.reindex(columns=all_columns)
final_test = test_merged.reindex(columns=all_columns)

# Save
output_dir = Path('data/processed')

train_path = output_dir / f"final_train.csv"
validate_path = output_dir / f"final_validate.csv"
test_path = output_dir / f"final_test.csv"

final_train.to_csv(train_path, index=False)
final_validate.to_csv(validate_path, index=False)
final_test.to_csv(test_path, index=False)
