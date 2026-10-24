import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -694446.2025743227
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")),
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")),
    XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
