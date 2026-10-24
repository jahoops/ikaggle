import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Train your Random Forest model
rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)

# Generate predictions from each tree
tree_preds = np.array([tree.predict(X_test) for tree in rf.estimators_])
pred_mean = np.mean(tree_preds, axis=0)
pred_std = np.std(tree_preds, axis=0)

# Display confidence intervals
lower_bound = pred_mean - 1.96 * pred_std
upper_bound = pred_mean + 1.96 * pred_std
print(f'Predicted Mean: {pred_mean}')
print(f'95% Confidence Interval: ({lower_bound}, {upper_bound})')
