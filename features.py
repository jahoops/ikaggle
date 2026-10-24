import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE, SelectFromModel, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from scipy import stats

# Load your data
data = pd.read_csv('data/processed/cleaned_train.csv')

# Correlation matrix
corr_matrix = data.corr()
# Get the correlation values with the target variable 'PremiumAmount'
target_corr = corr_matrix['PremiumAmount'].abs().sort_values(ascending=False)

# Prepare the data
X = data.drop('PremiumAmount', axis=1)
y = data['PremiumAmount']

# Use lower correlation threshold and p-value significance
correlation_threshold = 0.01  # 1% correlation
p_values = pd.DataFrame(index=X.columns, columns=['p_value'])
for feature in X.columns:
    correlation, p_value = stats.pearsonr(X[feature], y)
    p_values.loc[feature] = p_value

# Select features with either meaningful correlation or statistical significance
selected_features = target_corr[
    (target_corr > correlation_threshold) | 
    (p_values['p_value'] < 0.05)
].index.tolist()
print("Selected features based on correlation and p-value significance:", selected_features)

# Define the model
model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=50, max_depth=10)  # Speed up with fewer trees and max depth

# Perform RFE with percentage of features instead of 'all'
n_features = X.shape[1]
rfe = RFE(
    estimator=model,
    n_features_to_select=int(n_features * 0.3),  # Select top 30% of features
    step=5
)
rfe.fit(X, y)

# Filter RFE features by importance
rfe_importances = pd.Series(
    rfe.estimator_.feature_importances_[rfe.support_], 
    index=X.columns[rfe.support_]
)
selected_rfe_features = rfe_importances[rfe_importances > 0.01].index.tolist()
print("Selected features based on RFE (with importance threshold):", selected_rfe_features)

# Train the initial model
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Use variance ratio for feature selection
importance_mean = feature_importances['Importance'].mean()
importance_std = feature_importances['Importance'].std()
significance_threshold = importance_mean - (importance_std * 0.5)  # Less stringent threshold

# Select features that explain variance
top_features = feature_importances[
    feature_importances['Importance'] > significance_threshold
]['Feature'].tolist()

print("\nFeature Selection Summary:")
print(f"Total features analyzed: {len(X.columns)}")
print(f"Features with significant correlation: {len(selected_features)}")
print(f"Features selected by RFE: {len(selected_rfe_features)}")
print(f"Features with significant importance: {len(top_features)}")

# Combine all significant features
significant_features = pd.DataFrame({
    'Feature': list(set(selected_features + selected_rfe_features + top_features)),
    'Correlation': [abs(target_corr[f]) if f in target_corr else 0 for f in set(selected_features + selected_rfe_features + top_features)],
    'RF_Importance': [feature_importances.loc[feature_importances['Feature'] == f, 'Importance'].iloc[0] if f in feature_importances['Feature'].values else 0 for f in set(selected_features + selected_rfe_features + top_features)],
    'P_Value': [p_values.loc[f, 'p_value'] if f in p_values.index else 1.0 for f in set(selected_features + selected_rfe_features + top_features)]
})

# Sort by combined significance
significant_features['Combined_Score'] = (
    significant_features['Correlation'] * 0.3 +
    significant_features['RF_Importance'] * 0.5 +
    (1 - significant_features['P_Value']) * 0.2
)
significant_features = significant_features.sort_values('Combined_Score', ascending=False)

print("\nFeatures Ranked by Overall Significance:")
print(significant_features.to_string(index=False))

# Visualize all significant features
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))
plt.barh(significant_features['Feature'], significant_features['Combined_Score'])
plt.xlabel('Combined Significance Score')
plt.title('Feature Significance Analysis')
plt.tight_layout()
plt.savefig('feature_significance.png')
plt.show()  # Add this line to display the plot

# Additional visualization for better understanding
plt.figure(figsize=(15, 10))
sns.heatmap(
    significant_features[['Correlation', 'RF_Importance', 'P_Value', 'Combined_Score']], 
    annot=True, 
    cmap='YlOrRd',
    yticklabels=significant_features['Feature']
)
plt.title('Feature Significance Metrics Heatmap')
plt.tight_layout()
plt.savefig('feature_significance_heatmap.png')
plt.show()  # Show the second plot

# Save detailed feature analysis
significant_features.to_csv('feature_analysis.csv', index=False)

print("\nDetailed feature analysis saved to 'feature_analysis.csv'")
print("Feature significance visualization saved to 'feature_significance.png'")

# Use Lasso for feature selection
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X, y)
selector = SelectFromModel(lasso, threshold='mean', prefit=True)  # Use mean importance as threshold
selected_lasso_features = X.columns[selector.get_support()].tolist()
print("Selected features based on Lasso:", selected_lasso_features)

# Combine features from different methods
combined_features = set(selected_features) | set(selected_rfe_features) | set(top_features) | set(selected_lasso_features)
print("Final selected features:", combined_features)
