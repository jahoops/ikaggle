import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from typing import Dict, List, Tuple
import json
import joblib
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def safe_rmsle(y_true, y_pred):
    """Calculate RMSLE with protection against edge cases"""
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Find minimum positive value in true values
    min_val = max(1e-6, float(y_true[y_true > 0].min() if (y_true > 0).any() else 1e-6))
    
    # Clip predictions and true values
    y_pred_clipped = np.clip(y_pred, min_val, None)
    y_true_clipped = np.clip(y_true, min_val, None)
    
    # Safely calculate log values
    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)
    
    # Calculate squared differences
    squared_diff = np.square(log_pred - log_true)
    
    # Handle any remaining NaN values
    valid_mask = ~np.isnan(squared_diff)
    if not valid_mask.any():
        return float('inf')
    
    return np.sqrt(np.mean(squared_diff[valid_mask]))

class EnsembleSelector:
    def __init__(self):
        self.model = lgb.LGBMRegressor(
            objective='regression_l1',  # Keep MAE objective
            boosting_type='gbdt',
            n_estimators=2000,         # More trees for better convergence
            learning_rate=0.005,       # Smaller learning rate
            num_leaves=15,
            max_depth=4,
            min_data_in_leaf=100,      # Increased to prevent overfitting
            feature_fraction=0.8,      # Increased feature fraction
            bagging_fraction=0.8,      # Increased bagging fraction
            bagging_freq=5,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=1
        )
        self.binary_columns: List[str] = []
        self.n_extra_features = 4  # Number of engineered features (mean, std, min, max)
        self.use_log_transform = True

    def prepare_data(self, bitbins_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Memory-efficient data preparation with simpler features"""
        if not hasattr(self, 'initialized'):
            print("\nInitializing feature columns...")
            self.binary_columns = sorted([col for col in bitbins_data.columns if '_bin_' in col])
            self.submission_cols = [f'submission_{i}' for i in range(1, 7)]
            
            # Create weights for submissions based on global performance
            if 'PremiumAmount' in bitbins_data.columns:
                scores = []
                for col in self.submission_cols:
                    score = safe_rmsle(bitbins_data['PremiumAmount'], bitbins_data[col])
                    scores.append(score)
                self.submission_weights = 1.0 / (np.array(scores) + 1e-6)
                self.submission_weights /= self.submission_weights.sum()
                print("\nSubmission Weights:")
                for i, weight in enumerate(self.submission_weights, 1):
                    print(f"Submission {i}: {weight:.4f}")
            
            self.initialized = True

        # Process in chunks to reduce memory usage
        chunk_size = 10000
        feature_chunks = []
        
        for start_idx in range(0, len(bitbins_data), chunk_size):
            end_idx = min(start_idx + chunk_size, len(bitbins_data))
            chunk = bitbins_data.iloc[start_idx:end_idx]
            
            # Create features for this chunk
            chunk_features = pd.DataFrame()
            
            # Add binary features - but limit to most important ones
            if hasattr(self, 'important_bins'):
                chunk_features = chunk[self.important_bins].copy()
            else:
                chunk_features = chunk[self.binary_columns].copy()
            
            # Add submission predictions
            for col in self.submission_cols:
                chunk_features[col] = chunk[col]
            
            # Calculate ensemble predictions for chunk
            submissions = chunk_features[self.submission_cols].values
            weighted_pred = (submissions * self.submission_weights).sum(axis=1)
            
            # Add only essential features
            chunk_features['weighted_ensemble'] = weighted_pred
            
            # Convert to numpy and append
            feature_chunks.append(chunk_features.values)
            
            # Clear memory
            del chunk_features
            
        # Combine chunks
        X = np.vstack(feature_chunks)
        
        # Clear memory
        del feature_chunks
        
        if 'PremiumAmount' in bitbins_data.columns:
            return X, bitbins_data['PremiumAmount'].values
        return X, None

    def train(self, bitbins_data: pd.DataFrame):
        """Train with more detailed metrics"""
        X, y = self.prepare_data(bitbins_data)
        if y is None:
            raise ValueError("Training data must include PremiumAmount column")
        
        # Clip target values before log transform
        y = np.clip(y, 1e-6, 1e9)
        y = np.log1p(y)
        
        # Train initial model to identify important features
        print("\nTraining initial model for feature selection...")
        self.model.fit(X, y)
        
        # Select important binary features
        if not hasattr(self, 'important_bins'):
            feature_names = (
                self.binary_columns +
                self.submission_cols +
                ['weighted_ensemble']
            )
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Keep only binary features with non-zero importance
            self.important_bins = [
                col for col in self.binary_columns 
                if importance_df.loc[importance_df['feature'] == col, 'importance'].iloc[0] > 0
            ]
            print(f"\nSelected {len(self.important_bins)} important binary features")
        
        # Retrain with selected features
        X, y = self.prepare_data(bitbins_data)  # Now uses important_bins
        
        # Use stratified k-fold to handle data distribution
        n_folds = 5
        fold_scores = []
        
        print(f"\nPerforming {n_folds}-fold cross-validation...")
        for fold in range(n_folds):
            # Create fold indices
            all_idx = np.arange(len(X))
            val_idx = all_idx[fold::n_folds]
            train_idx = np.setdiff1d(all_idx, val_idx)
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Custom eval metric for lightgbm
            def rmsle_eval(y_pred, dataset):
                y_true = dataset.get_label()
                y_pred = np.clip(np.expm1(y_pred), 1e-6, 1e9)
                y_true = np.clip(np.expm1(y_true), 1e-6, 1e9)
                return 'RMSLE', safe_rmsle(y_true, y_pred), False

            print("\nTraining with early stopping...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                eval_metric=[rmsle_eval, 'l1'],  # Track both RMSLE and MAE
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100),  # Increased patience
                    lgb.log_evaluation(period=50),
                    lgb.record_evaluation()  # Record metrics for analysis
                ]
            )
            
            # Print final performance
            best_iteration = self.model.best_iteration_
            print(f"\nBest iteration: {best_iteration}")
            print("Final metrics:")
            for metric, values in self.model.evals_result_.items():
                for dataset, scores in values.items():
                    print(f"{metric} on {dataset}: {scores[-1]:.6f}")

            # Score fold with safe prediction
            val_pred = self.predict(bitbins_data.iloc[val_idx])
            fold_rmsle = safe_rmsle(y_val, val_pred)
            
            if np.isfinite(fold_rmsle):  # Only add valid scores
                fold_scores.append(fold_rmsle)
                print(f"Fold {fold + 1} RMSLE: {fold_rmsle:.6f}")
            else:
                print(f"Fold {fold + 1} RMSLE: Invalid score, skipping")
        
        if fold_scores:
            print(f"\nCross-validation RMSLE: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
        else:
            print("\nWarning: No valid fold scores obtained")
        
        # Final training on full dataset
        print("\nTraining final model on full dataset...")
        self.model.fit(X, y)

    def analyze_errors(self, features: pd.DataFrame) -> pd.DataFrame:
        """Analyze prediction errors to identify patterns"""
        predictions = self.predict(features)
        if 'PremiumAmount' in features.columns:
            errors = np.abs(np.log1p(predictions) - np.log1p(features['PremiumAmount']))
            error_analysis = pd.DataFrame({
                'id': features['id'],
                'actual': features['PremiumAmount'],
                'predicted': predictions,
                'error': errors
            })
            return error_analysis.sort_values('error', ascending=False)
        return None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict with numerical stability safeguards"""
        X, _ = self.prepare_data(features)
        
        # Get raw predictions in log space
        log_predictions = self.model.predict(X)
        
        # Clip predictions to avoid overflow
        log_predictions = np.clip(log_predictions, -10, 10)
        
        # Safely transform back from log space
        predictions = np.expm1(log_predictions)
        
        # Ensure valid range
        predictions = np.clip(predictions, 1e-6, 1e9)
        
        return predictions

    def save_predictions(self, features: pd.DataFrame, output_path: Path):
        """Save predictions in submission format"""
        predictions = self.predict(features)
        submission_df = pd.DataFrame({
            'id': features['id'],
            'Premium Amount': predictions
        })
        submission_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")

    def analyze_submissions(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze the performance and contribution of each submission"""
        if 'PremiumAmount' not in data.columns:
            raise ValueError("Cannot analyze submissions without actual values")

        results = []
        baseline_pred = self.predict(data)
        baseline_rmsle = safe_rmsle(data['PremiumAmount'], baseline_pred)

        print("\nAnalyzing Submission Contributions:")
        print(f"Baseline Ensemble RMSLE: {baseline_rmsle:.6f}")
        print("\nIndividual Performance:")

        # Individual performance
        for i, col in enumerate(self.submission_cols, 1):
            sub_rmsle = safe_rmsle(data['PremiumAmount'], data[col])
            weight = self.submission_weights[i-1]
            results.append({
                'submission_id': i,
                'rmsle': sub_rmsle,
                'weight': weight,
                'contribution': 0  # Will be updated after leave-one-out
            })
            print(f"Submission {i}: RMSLE={sub_rmsle:.6f}, Weight={weight:.4f}")

        print("\nLeave-one-out Impact:")
        
        # Leave-one-out analysis
        for i, col in enumerate(self.submission_cols, 1):
            # Create temporary data without this submission
            temp_data = data.copy()
            temp_data[col] = temp_data[self.submission_cols].mean(axis=1)  # Replace with mean
            
            # Get predictions without this submission
            leave_one_out_pred = self.predict(temp_data)
            leave_one_out_rmsle = safe_rmsle(data['PremiumAmount'], leave_one_out_pred)
            
            # Calculate impact
            impact = leave_one_out_rmsle - baseline_rmsle
            results[i-1]['contribution'] = impact
            
            print(f"Submission {i} Impact: {impact:.6f} " + 
                  ("(Helpful) ✓" if impact > 0 else "(Not Helpful) ✗"))

        # Create summary DataFrame
        analysis_df = pd.DataFrame(results).sort_values('contribution', ascending=False)
        
        print("\nRecommendations:")
        print("Keep submissions with positive contribution and higher weights")
        keep_subs = analysis_df[analysis_df['contribution'] > 0]['submission_id'].tolist()
        print(f"Recommended submissions to keep: {keep_subs}")
        
        return analysis_df

    def fit(self, X, y):
        # Transform target variable if enabled
        if self.use_log_transform:
            y_transformed = np.log1p(y)  # log1p handles zero values
        else:
            y_transformed = y

        # Perform cross-validation for each model
        self.cv_scores_ = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y_transformed, 
                                  cv=5, scoring='neg_mean_squared_error')
            self.cv_scores_[name] = -scores.mean()  # Convert to positive MSE

        # Update feature importance with transformed target
        for name, model in self.models.items():
            model.fit(X, y_transformed)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance_[name] = model.feature_importances_

        # Select best model based on cross-validation
        self.best_model_ = min(self.cv_scores_, key=self.cv_scores_.get)
        self.model = self.models[self.best_model_]
        
        # Final fit with best model
        self.model.fit(X, y_transformed)
        return self

def main():
    # Load data
    bitbins_path = Path('/home/USER/Projects/ikaggle/AAA/bitbins/train_bitbins_scored.csv')
    bitbins_data = pd.read_csv(bitbins_path)
    
    # Add submission predictions to bitbins_data
    with open('/home/USER/Projects/ikaggle/AAA/submissions.JSON', 'r') as f:
        submissions = json.load(f)['submissions']
    
    for submission in submissions:
        sub_id = submission['id']
        sub_data = pd.read_csv(submission['train'])
        bitbins_data[f'submission_{sub_id}'] = pd.merge(
            bitbins_data[['id']], 
            sub_data[['id', 'Premium Amount']], 
            on='id', 
            how='left'
        )['Premium Amount']
    
    # Create and train selector
    selector = EnsembleSelector()

    # Save bitbins_data to a file
    bitbins_path = Path('/home/USER/Projects/ikaggle/AAA/bitbins/bitbin_train_data.csv')
    bitbins_data.to_csv(bitbins_path, index=False)
    print(f"Saved bitbins data to {bitbins_path}")

    selector.train(bitbins_data)
    
    # Analyze submissions
    analysis_df = selector.analyze_submissions(bitbins_data)
    
    # Save analysis
    analysis_path = model_dir / 'submission_analysis.csv'
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\nSubmission analysis saved to: {analysis_path}")
    
    # Make predictions directly
    predictions = selector.predict(bitbins_data)
    
    # Calculate overall RMSLE with safe function
    actual = bitbins_data['PremiumAmount']
    rmsle = safe_rmsle(actual, predictions)
    print(f"\nFinal RMSLE: {rmsle:.6f}")
    
    # Save predictions
    output_dir = Path('/home/USER/Projects/ikaggle/AAA/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    selector.save_predictions(bitbins_data, output_dir / 'ensemble_predictions.csv')
    
    # Create models directory if it doesn't exist
    model_dir = Path('/home/USER/Projects/ikaggle/AAA/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / 'ensemble_selector.lgb'
    joblib.dump(selector, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Get correct feature names
    feature_names = (
        selector.binary_columns +  # Binary features
        selector.submission_cols +  # Raw submission predictions
        ['weighted_ensemble', 'best_submission'] +  # Ensemble features
        [f'deviation_{i+1}' for i in range(3)]  # Only 3 deviation features were created
    )
    
    # Get feature importances
    importances = selector.model.feature_importances_
    
    # Ensure lengths match
    if len(feature_names) != len(importances):
        print(f"\nWarning: Feature name length ({len(feature_names)}) "
              f"doesn't match importance length ({len(importances)})")
        # Use only the number of features we actually have
        feature_names = feature_names[:len(importances)]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save and display
    importance_df.to_csv(model_dir / 'feature_importance.csv', index=False)
    print("\nFeature Importance Summary:")
    print(importance_df.head(10))

    # Create SVR with regularization and adjusted parameters
    svr = SVR(
        kernel='rbf',  # Radial basis function kernel
        C=1.0,         # Lower C value for stronger regularization
        epsilon=0.1,   # Wider margin around regression line
        gamma='scale'  # Automatically scale gamma based on features
    )

    # Optional: Use GridSearchCV to find optimal parameters
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    grid_search = GridSearchCV(
        svr, param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the model
    X_scaled = StandardScaler().fit_transform(X)
    grid_search.fit(X_scaled, y)

    # Use best parameters
    best_svr = grid_search.best_estimator_

    if __name__ == "__main__":
        main()
