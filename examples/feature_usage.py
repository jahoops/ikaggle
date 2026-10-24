from features.optimal_features import OptimalFeatureSelector

# Initialize selector
selector = OptimalFeatureSelector()

# Transform features
X_transformed = selector.transform(X)

# Get feature importance with transformed features
model.fit(X_transformed, y)
importances = pd.DataFrame({
    'Feature': X_transformed.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 20 Most Important Features:")
print(importances.head(20))
