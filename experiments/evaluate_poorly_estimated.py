import os
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the best model
model_path = os.path.join("output", "Svr", "model", "best_model.joblib")
model = joblib.load(model_path)

# Load the validation data
data_path = os.path.join("data", "processed", "final_validate.csv")
data = pd.read_csv(data_path)

target_column = "PremiumAmount"
X = data.drop(target_column, axis=1)
y_true = data[target_column]

# Make predictions and compute absolute errors
y_pred = model.predict(X)
errors = np.abs(y_pred - y_true)

# Determine the threshold for poorly estimated samples (top 10% error)
error_threshold = np.percentile(errors, 90)
poor_samples_mask = errors >= error_threshold
poor_samples = data[poor_samples_mask].copy()

# Select a sample with the maximum error for explanation
idx = errors.argmax()
sample = X.iloc[[idx]]  # must be 2D for both explainers

# ---- SHAP Explanation ----
# Using a KernelExplainer (model-agnostic) with a background sample
# For background we take 100 random samples from the full dataset
background = X.sample(n=100, random_state=42)
explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(sample, nsamples=100)

# Plot the SHAP explanation for the selected sample
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, sample, matplotlib=True)
plt.title("SHAP Explanation for Poorly Estimated Sample")
plt.show()

# Additional SHAP plots
shap.summary_plot(shap_values, sample)
# or for a bar plot:
shap.bar_plot(shap_values, feature_names=sample.columns)

# ---- LIME Explanation ----
# Prepare a LimeTabularExplainer using the full feature set and providing a training set sample as background.
explainer_lime = LimeTabularExplainer(
    training_data=np.array(X),
    feature_names=X.columns,
    mode='regression',
    verbose=True,
    random_state=42
)

# Generate explanation for the selected sample
lime_exp = explainer_lime.explain_instance(
    data_row=sample.iloc[0],
    predict_fn=model.predict,
    num_features=len(X.columns)
)

# Show LIME explanation in text form
print(lime_exp.as_list())

# Optionally, display the LIME explanation in a notebook-friendly html format:
lime_exp.save_to_file("lime_explanation.html")
print("LIME explanation saved to lime_explanation.html")