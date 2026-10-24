import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

# Create pipeline
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=3, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")),
    StackingEstimator(estimator=XGBRegressor(colsample_bytree=0.8, learning_rate=0.01, max_depth=5, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")),
    XGBRegressor(colsample_bytree=0.8, learning_rate=0.1, max_depth=3, n_estimators=100, random_state=42, subsample=0.8, tree_method="gpu_hist")
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)
