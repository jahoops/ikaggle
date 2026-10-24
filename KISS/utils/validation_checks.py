import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json

class PipelineValidator:
    """Tool to check data integrity at each step of the pipeline"""
    
    def __init__(self, checkpoint_dir='/home/USER/Projects/ikaggle/KISS/checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_count = 0
        self.errors_found = 0
        self.warnings_found = 0
        self.log = []
        
    def check_data(self, data, name="unnamed_data", stage="unknown", expectations=None):
        """Check a DataFrame or array for issues"""
        self.checkpoint_count += 1
        checkpoint_id = f"checkpoint_{self.checkpoint_count:03d}"
        checkpoint_path = f"{self.checkpoint_dir}/{checkpoint_id}_{stage}_{name}"
        
        print(f"\n==== VALIDATION CHECKPOINT {self.checkpoint_count} ====")
        print(f"Stage: {stage}")
        print(f"Data: {name}")
        
        # Convert array to dataframe if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                # 1D array - convert to Series
                data_df = pd.DataFrame({name: data})
            else:
                # 2D array - convert to DataFrame with generic column names
                columns = [f"col_{i}" for i in range(data.shape[1])]
                data_df = pd.DataFrame(data, columns=columns)
                
            # Save checkpoint as CSV
            data_df.to_csv(f"{checkpoint_path}.csv", index=False)
        elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            # Save checkpoint as CSV
            data_df = data
            data.to_csv(f"{checkpoint_path}.csv", index=False)
        else:
            print(f"Warning: Unsupported data type: {type(data)}")
            self.warnings_found += 1
            return False
        
        # Get basic statistics
        stats = {}
        warnings = []
        errors = []
        
        try:
            if isinstance(data_df, pd.DataFrame):
                # Get stats for each column
                for col in data_df.columns:
                    col_data = data_df[col].values
                    stats[col] = self._get_column_stats(col_data)
                    
                    # Check for issues
                    col_warnings, col_errors = self._check_column(col_data, col, expectations)
                    warnings.extend(col_warnings)
                    errors.extend(col_errors)
                    
            else:  # Series
                values = data_df.values
                stats["values"] = self._get_column_stats(values)
                
                # Check for issues
                col_warnings, col_errors = self._check_column(values, name, expectations)
                warnings.extend(col_warnings)
                errors.extend(col_errors)
        except Exception as e:
            print(f"Error analyzing data: {e}")
            errors.append(f"Error analyzing data: {e}")
        
        # Create visualization if numeric
        try:
            if isinstance(data_df, pd.DataFrame):
                numeric_cols = data_df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(12, 8))
                    for i, col in enumerate(numeric_cols[:min(9, len(numeric_cols))]):
                        plt.subplot(3, 3, i+1)
                        sns.histplot(data_df[col], kde=True)
                        plt.title(col)
                    plt.tight_layout()
                    plt.savefig(f"{checkpoint_path}_histograms.png")
                    plt.close()
            else:  # Series
                if np.issubdtype(data_df.dtype, np.number):
                    plt.figure(figsize=(8, 6))
                    sns.histplot(data_df, kde=True)
                    plt.title(name)
                    plt.savefig(f"{checkpoint_path}_histogram.png")
                    plt.close()
        except Exception as e:
            print(f"Error creating visualization: {e}")
            warnings.append(f"Error creating visualization: {e}")
        
        # Log issues
        self.warnings_found += len(warnings)
        self.errors_found += len(errors)
        
        # Print summary
        print(f"\nStatistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for stat_name, stat_value in value.items():
                    print(f"    {stat_name}: {stat_value}")
            else:
                print(f"  {key}: {value}")
                
        if warnings:
            print(f"\nWARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"  ⚠️ {warning}")
                
        if errors:
            print(f"\nERRORS ({len(errors)}):")
            for error in errors:
                print(f"  ❌ {error}")
                
        log_entry = {
            "checkpoint": self.checkpoint_count,
            "stage": stage,
            "name": name,
            "timestamp": pd.Timestamp.now().isoformat(),
            "stats": stats,
            "warnings": warnings,
            "errors": errors,
        }
        
        self.log.append(log_entry)
        
        # Save log
        with open(f"{self.checkpoint_dir}/validation_log.json", 'w') as f:
            json.dump(self.log, f, indent=2)
        
        # Return result
        return len(errors) == 0
    
    def _get_column_stats(self, data):
        """Get basic statistics for a column"""
        stats = {}
        try:
            stats["count"] = len(data)
            stats["missing"] = int(np.isnan(data).sum()) if np.issubdtype(data.dtype, np.number) else 0
            
            # Only calculate stats for numeric data with valid values
            if np.issubdtype(data.dtype, np.number) and len(data) > stats["missing"]:
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    stats["min"] = float(np.min(valid_data))
                    stats["max"] = float(np.max(valid_data))
                    stats["mean"] = float(np.mean(valid_data))
                    stats["median"] = float(np.median(valid_data))
                    stats["std"] = float(np.std(valid_data))
                    stats["zero_count"] = int(np.sum(valid_data == 0))
                    stats["zero_percent"] = float(100 * stats["zero_count"] / len(valid_data))
                    stats["negative_count"] = int(np.sum(valid_data < 0))
                    stats["negative_percent"] = float(100 * stats["negative_count"] / len(valid_data))
        except Exception as e:
            stats["error"] = str(e)
            
        return stats
    
    def _check_column(self, data, col_name, expectations=None):
        """Check a column for issues"""
        warnings = []
        errors = []
        
        # Skip non-numeric columns
        if not np.issubdtype(data.dtype, np.number):
            return warnings, errors
        
        # Basic checks for missing and infinite values
        if np.isnan(data).any():
            warnings.append(f"Column '{col_name}' has {np.isnan(data).sum()} missing values")
            
        if np.isinf(data).any():
            errors.append(f"Column '{col_name}' has {np.isinf(data).sum()} infinite values")
            
        valid_data = data[~np.isnan(data) & ~np.isinf(data)]
        
        # Define insurance-specific features where high zero counts are expected
        zero_tolerant_features = [
            'PreviousClaims', 'ClaimsPerYear', 'Age_PreviousClaims_Interaction',
            'InsuranceDuration_PreviousClaims_Interaction', 'HealthScore_SmokingStatus_Interaction'
        ]
        
        # Define binary features that should be 0-1 only
        binary_features = ['SmokingStatus', 'Gender', 'ExerciseFrequency']
        
        # Zero checks - with special treatment for claim-related features
        zero_percent = 100 * np.sum(valid_data == 0) / len(valid_data) if len(valid_data) > 0 else 0
        
        # Check if this is a claims-related feature with expected zeros
        is_claim_feature = any(term in col_name for term in zero_tolerant_features) or 'Claims' in col_name
        
        # Be less strict with zeros for claim features (up to 70% is ok)
        if not is_claim_feature:
            if zero_percent > 20:
                warnings.append(f"Column '{col_name}' has {zero_percent:.1f}% zero values")
            if zero_percent > 40:
                errors.append(f"Column '{col_name}' has too many zeros ({zero_percent:.1f}%)")
        else:
            # For claims data, only flag if extremely high
            if zero_percent > 90:
                warnings.append(f"Column '{col_name}' has extreme zero values ({zero_percent:.1f}%)")
                
        # Negative checks - with special handling for binary features
        neg_percent = 100 * np.sum(valid_data < 0) / len(valid_data) if len(valid_data) > 0 else 0
        
        # Binary features should never be negative
        is_binary = any(term in col_name for term in binary_features)
        
        # Also consider scaled features that are allowed to be negative
        is_scaled_feature = any(term in col_name.lower() for term in 
                               ['_scaled', 'interaction', 'ratio', 'date_', 'day', 'month', 'year'])
        
        # Binary features should never be negative
        if is_binary and neg_percent > 0:
            errors.append(f"Binary column '{col_name}' has {neg_percent:.1f}% negative values")
        elif not is_scaled_feature and neg_percent > 0:
            warnings.append(f"Column '{col_name}' has {neg_percent:.1f}% negative values")
            if neg_percent > 20:
                errors.append(f"Column '{col_name}' has too many negative values ({neg_percent:.1f}%)")
        
        # Check specific expectations if provided
        if expectations is not None and col_name in expectations:
            col_expectations = expectations[col_name]
            
            if "min" in col_expectations and min(valid_data) < col_expectations["min"]:
                errors.append(f"Column '{col_name}' minimum ({min(valid_data)}) is below expected minimum ({col_expectations['min']})")
                
            if "max" in col_expectations and max(valid_data) > col_expectations["max"]:
                errors.append(f"Column '{col_name}' maximum ({max(valid_data)}) is above expected maximum ({col_expectations['max']})")
                
            if "mean_range" in col_expectations:
                mean = np.mean(valid_data)
                min_mean, max_mean = col_expectations["mean_range"]
                if mean < min_mean or mean > max_mean:
                    errors.append(f"Column '{col_name}' mean ({mean}) is outside expected range ({min_mean}, {max_mean})")
        
        return warnings, errors
    
    def summarize(self):
        """Print a summary of all validations"""
        print("\n==== VALIDATION SUMMARY ====")
        print(f"Total checkpoints: {self.checkpoint_count}")
        print(f"Warnings found: {self.warnings_found}")
        print(f"Errors found: {self.errors_found}")
        
        if self.errors_found > 0:
            print("\nErrors detected! Check the validation log for details.")
            return False
        else:
            print("\nAll validations passed successfully.")
            return True

    def check_data_quality(self, X_train, y_train, X_test=None):
        """
        Check for data quality issues.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            
        Returns:
            Dictionary with quality check results
        """
        results = {}
        
        # Check for missing values
        train_missing = X_train.isnull().sum().sum()
        results['train_missing_values'] = train_missing
        
        if X_test is not None:
            test_missing = X_test.isnull().sum().sum()
            results['test_missing_values'] = test_missing
        
        # Check for invalid target values
        negative_targets = (y_train < 0).sum()
        zero_targets = (y_train == 0).sum()
        results['negative_targets'] = negative_targets
        results['zero_targets'] = zero_targets
        
        # Check feature distributions
        results['feature_stats'] = X_train.describe()
        
        # Print results
        print("Data Quality Check Results:")
        print(f"Train Missing Values: {train_missing}")
        if X_test is not None:
            print(f"Test Missing Values: {test_missing}")
        print(f"Negative Target Values: {negative_targets}")
        print(f"Zero Target Values: {zero_targets}")
        
        return results
        
    def validate_predictions(self, y_true=None, y_pred=None, threshold=5.0):
        """
        Check if predictions are valid for insurance premiums.
        
        Args:
            y_true: True values (if available)
            y_pred: Predicted values
            threshold: Maximum allowed ratio for extreme predictions
            
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        if y_pred is None:
            return results
            
        # Check for negative predictions
        negative_preds = (y_pred < 0).sum()
        results['negative_predictions'] = negative_preds
        
        # Check for zero predictions
        zero_preds = (y_pred == 0).sum()
        results['zero_predictions'] = zero_preds
        
        # Check for extreme predictions
        if y_true is not None:
            # Identify extreme predictions
            ratio = np.abs(y_pred / (y_true + 1e-10))
            extreme_preds = (ratio > threshold).sum()
            results['extreme_predictions'] = extreme_preds
            results['max_ratio'] = ratio.max()
            
            # Compute error metrics
            mae = np.mean(np.abs(y_pred - y_true))
            mse = np.mean((y_pred - y_true)**2)
            rmse = np.sqrt(mse)
            
            results['mae'] = mae
            results['mse'] = mse
            results['rmse'] = rmse
            
            print("Prediction Validation Results:")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"Extreme Predictions (ratio > {threshold}): {extreme_preds}")
        
        # Print general results
        print(f"Negative Predictions: {negative_preds}")
        print(f"Zero Predictions: {zero_preds}")
        
        return results
        
    def plot_predictions_vs_actual(self, y_true, y_pred, log_scale=True, figsize=(10, 6)):
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            log_scale: Whether to use log scale for axes
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Set labels and title
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        
        # Set log scale if specified
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
            
        plt.grid(alpha=0.3)
        
        return plt.gcf()
        
    def plot_error_distribution(self, y_true, y_pred, figsize=(10, 6)):
        """
        Plot distribution of prediction errors.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        errors = y_pred - y_true
        
        plt.figure(figsize=figsize)
        
        # Plot error histogram
        plt.hist(errors, bins=50, alpha=0.7)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--')
        
        # Set labels and title
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Distribution of Prediction Errors")
        plt.grid(alpha=0.3)
        
        return plt.gcf()
