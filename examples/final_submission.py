from features.optimal_features import OptimalFeatureSelector
from features.submission_features import SubmissionFeatureSelector
import pandas as pd

# Load and prepare data
train_data = pd.read_csv('../data/processed/cleaned_train.csv')
test_data = pd.read_csv('../data/processed/cleaned_test.csv')

# Create initial features
feature_creator = OptimalFeatureSelector()
X_train = feature_creator.transform(train_data)
y_train = train_data['PremiumAmount']
X_test = feature_creator.transform(test_data)

# Analyze feature importance
submission_selector = SubmissionFeatureSelector(importance_threshold=0.01)
feature_groups = submission_selector.analyze_features(X_train, y_train)

# Print detailed feature analysis
submission_selector.print_feature_report()

# Transform data using only important features
X_train_final = submission_selector.transform_for_submission(X_train)
X_test_final = submission_selector.transform_for_submission(X_test)

print("\nFinal feature counts:")
print(f"Original features: {X_train.shape[1]}")
print(f"Selected features: {X_train_final.shape[1]}")

# Save feature importance analysis
feature_importance_df = submission_selector.feature_importances
feature_importance_df.to_csv('../data/feature_importance_analysis.csv', index=False)
