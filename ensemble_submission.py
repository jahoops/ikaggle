import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_models(output_dir):
    """Load XGBoost, NeuralNet, and CatBoost models"""
    models = {}
    model_paths = {
        'XGBoost': Path(output_dir) / 'v1' / 'XGBoost' / 'best_XGBoost_model.pkl',
        'NeuralNet': Path(output_dir) / 'v1' / 'NeuralNet' / 'best_NeuralNet_model.pkl',
        'CatBoost': Path(output_dir) / 'v1' / 'CatBoost' / 'best_CatBoost_model.pkl',
    }
    
    for name, path in model_paths.items():
        if path.exists():
            try:
                model_info = joblib.load(path)
                models[name] = model_info['model']  # Extract the actual model object
                logger.info(f"Loaded model: {name} from {path}")
            except Exception as e:
                logger.error(f"Error loading model {name} from {path}: {str(e)}")
        else:
            logger.warning(f"Model file for {name} not found at {path}. Skipping this model.")
    
    if not models:
        logger.error("No models were loaded. Ensure that models are trained and saved correctly.")
        raise FileNotFoundError("No model files found for ensemble.")
    
    return models

def find_similar_records(test_record, train_data, n_neighbors=50):
    """Find similar records in training data"""
    scaler = StandardScaler()
    combined = pd.concat([train_data, test_record.to_frame().T], ignore_index=True)
    scaler.fit(combined)
    train_scaled = scaler.transform(train_data)
    test_scaled = scaler.transform(test_record.to_frame().T)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(train_scaled)
    distances, indices = nbrs.kneighbors(test_scaled)
    return train_data.iloc[indices[0]]

def determine_best_model(similar_train_records, models, train_targets):
    """Determine the best model based on RMSLE on similar records"""
    best_model = None
    lowest_rmsle = float('inf')
    
    for name, model in models.items():
        try:
            predictions = model.predict(similar_train_records)
            score = safe_rmsle(train_targets.loc[similar_train_records.index], predictions)
            if score < lowest_rmsle:
                lowest_rmsle = score
                best_model = model
        except Exception as e:
            logger.error(f"Error evaluating model {name}: {str(e)}")
            continue
    
    return best_model

def safe_rmsle(y_true, y_pred):
    """Calculate RMSLE safely"""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def generate_ensemble_predictions(output_dir, test_path, train_path, submission_path):
    """Generate ensemble predictions"""
    try:           
        test_data = pd.read_csv(test_path)
        train_data = pd.read_csv(train_path).drop(columns=['PremiumAmount'])
        train_targets = pd.read_csv(train_path)['PremiumAmount']
        
        models = load_models(output_dir)
        if not models:
            raise ValueError("No models loaded for ensemble.")
        
        all_predictions = []
        model_scores = []
        model_names = []
        
        # Load model performance report
        model_performance_path = Path(output_dir) / 'v1' / 'model_performance_report.json'
        if not model_performance_path.exists():
            logger.warning(f"Model performance report not found at {model_performance_path}. Proceeding without scores.")
            model_performance = {}
        else:
            with open(model_performance_path, 'r') as f:
                model_performance = json.load(f)
        
        # Extract model scores
        model_scores_dict = {}
        if 'summary' in model_performance:
            for model in model_performance['summary']:
                model_scores_dict[model['model']] = model['score']
        
        # Generate predictions from each model
        for model_name, model in models.items():
            try:
                logger.info(f"\nProcessing {model_name}...")
                features = model_info.get('features', [])  # Ensure features are available
                if not features:
                    logger.warning(f"No features found for model {model_name}. Skipping.")
                    continue
                
                # Get predictions
                X_test = test_data[features]
                pred = model.predict(X_test)
                pred = np.maximum(pred, 0)
                
                all_predictions.append(pred)
                model_names.append(model_name)
                model_scores.append(model_scores_dict.get(model_name, None))
                
                logger.info(f"{model_name} prediction stats:")
                logger.info(f"  Range: {pred.min():.2f} - {pred.max():.2f}")
                logger.info(f"  Mean: {pred.mean():.2f}")
                logger.info(f"  Median: {np.median(pred):.2f}")
                
            except Exception as e:
                logger.error(f"Error using {model_name}: {str(e)}")
                continue
        
        if not all_predictions:
            raise ValueError("No valid predictions generated")
        
        # Convert to array
        predictions = np.array(all_predictions)
        
        # Handle models with missing scores by assigning uniform weights
        model_scores = np.array([
            score if score is not None else np.mean(model_scores)
            for score in model_scores
        ])
        
        # Replace any remaining None with median score
        median_score = np.median(model_scores[np.isfinite(model_scores)])
        model_scores = np.where(
            np.isfinite(model_scores),
            model_scores,
            median_score
        )
        
        # Calculate weights inversely proportional to scores
        weights = 1 / (model_scores + 1e-8)
        weights /= np.sum(weights)
        
        # Generate weighted predictions
        weighted_preds = predictions * weights[:, np.newaxis]
        final_predictions = np.sum(weighted_preds, axis=0)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_data['id'],
            'PremiumAmount': final_predictions
        })
        
        # Save submission
        submission.to_csv(submission_path, index=False)
        logger.info(f"\nEnsemble submission saved to {submission_path}")
        logger.info(f"Final predictions:")
        logger.info(f"  Range: {final_predictions.min():.2f} - {final_predictions.max():.2f}")
        logger.info(f"  Mean: {final_predictions.mean():.2f}")
        logger.info(f"  Median: {np.median(final_predictions):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating submission: {str(e)}")
        return False

if __name__ == "__main__":
    output_directory = Path('/home/USER/Kaggle/InsuranceNext/output')
    test_csv = output_directory / 'v1' / 'cleaned_test.csv'
    train_csv = output_directory / 'v1' / 'cleaned_train.csv'
    submission_csv = output_directory / 'v1' / 'ensemble_submission.csv'
    
    generate_ensemble_predictions(output_directory, test_csv, train_csv, submission_csv)