# This file is for the meta predictor implementation. 
# The meta predictor is a model that uses the predictions of other models
# as features to make its own predictions. The SubmissionManager class is
# used to load and manage the submissions that will be used to train the
# meta predictor. The SubmissionManager loads submission information on init
# that includes data paths for training, validation, and test data. The
# get_submission method returns the submission data paths for a given ID.

from submission_manager import SubmissionManager
import pandas as pd
from meta_selector import MetaSelector

def load_data(file_path):
    data = pd.read_csv(file_path)   
    return data

# The MetaPredictor class is used to train and make predictions with the
# meta predictor model. The fit method trains the model using the
# predictions from the submissions as features and the actual target values
# as labels. The predict method makes predictions using the trained model.

# The save_model method saves the trained model to a file.

# The load_model method loads a trained model from a file.

# The main function in this file is used to train the meta predictor model
# and make predictions on the test data. The trained model is saved to a file.
# The main function also contains code to evaluate the model's performance
# on both validation and test datasets, ensuring generalization capabilities.
# The performance metrics used are RMSE and R2 score.

# The main function also contains code to generate a submission file for the
# test data predictions.

# The main function takes the following arguments:
# - submission_file: The path to the JSON file containing the submissions
# - train_file: The path to the training data file
# - validate_file: The path to the validation data file
# - test_file: The path to the test data file
# - model_file: The path to save the trained model
# - submission_output_file: The path to save the submission file

# The main function uses the SubmissionManager to load the submissions from
# the submission_file. It then loads the training, validation, and test data
# from the respective file paths. The MetaPredictor class is used to train the
# meta predictor model on the training data and make predictions on the
# validation and test data. The trained model is saved to the model_file.
# The predictions on the test data are saved to the submission_output_file.
# The performance of the model is evaluated on the validation and test data.
# After evaluation, the results are printed and the submission file is generated.
# This process includes saving performance metrics for later analysis.
# The main function is executed when the file is run as a script.

# The main function can be used to train and evaluate the meta predictor model
# on different datasets by providing the appropriate file paths as arguments.

# 1. Import the classes from each file:  
#    • BinManager (for creating bins)  
#    • MetaSelector (or other meta-model classes)  
#    • SubmissionManager (for handling submissions)

# 2. In meta_predictor.py, instantiate them in your main flow. For example:  
#    • Use SubmissionManager to get data paths or load submission data  
#    • Use BinManager to create or identify bins for features  
#    • Use MetaSelector to train a meta-model that picks the best submission or combines predictions

# 3. Pass data between these components as you build features, gather bins, and make meta-level decisions.
#    The exact details depend on your pipeline, but each file provides a specialized part of the flow.

from sklearn.metrics import mean_squared_error, mean_squared_log_error
import numpy as np
from sklearn.metrics import r2_score

def safe_rmsle(y_true, y_pred) -> float:
    """Calculate RMSLE with protection against negative values"""
    # Convert GPU arrays to numpy if needed
    if hasattr(y_true, 'to_numpy'):
        y_true = y_true.to_numpy()
    if hasattr(y_pred, 'to_numpy'):
        y_pred = y_pred.to_numpy()
        
    # Ensure arrays are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip values
    min_val = max(1e-6, float(y_true.min()))
    y_pred_clipped = np.clip(y_pred, min_val, None)
    y_true_clipped = np.clip(y_true, min_val, None)
    
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def main(submission_file, train_file, validate_file, test_file, model_file, submission_output_file):

    submission_manager = SubmissionManager(submission_file)
    submissions = submission_manager.load_submissions()

    train_data = load_data('/home/USER/Projects/ikaggle/data/train.csv')
    validate_data = load_data('/home/USER/Projects/ikaggle/data/validate.csv')
    test_data = load_data('/home/USER/Projects/ikaggle/data/test.csv')


    meta_selector = MetaSelector()
    meta_selector.fit(submissions, train_data)


    validate_predictions = meta_selector.predict(validate_data)

    validate_rmsle = safe_rmsle(validate_data, validate_predictions)
    validate_r2 = calculate_r2(validate_data, validate_predictions)

    test_predictions = meta_selector.predict(test_data)

    test_rmsle = safe_rmsle(test_data, test_predictions)
    test_r2 = calculate_r2(test_data, test_predictions)

    meta_selector.save_model(model_file)

