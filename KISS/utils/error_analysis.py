import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from utils.segmentation import InsuranceSegmenter

class ErrorAnalyzer:
    """Identifies and analyzes poorly predicted samples for additional segmentation."""
    
    def __init__(self, error_threshold=0.75):
        """
        Initialize the analyzer with a threshold for high error.
        
        Args:
            error_threshold: Percentile of error above which samples are considered 'poorly predicted'
        """
        self.error_threshold = error_threshold
        self.high_error_features = None
        self.error_segment_rules = None
    
    def compute_validation_errors(self, y_true, y_pred, X_val):
        """Compute errors on validation data and link them to features"""
        # Calculate errors (absolute or squared)
        errors = np.abs(y_true - y_pred)
        
        # Create a DataFrame with features and corresponding errors
        error_df = X_val.copy()
        error_df['prediction_error'] = errors
        
        # Determine high error threshold (e.g., top 25% of errors)
        high_error_threshold = np.percentile(errors, self.error_threshold * 100)
        error_df['is_high_error'] = error_df['prediction_error'] > high_error_threshold
        
        self.error_df = error_df
        print(f"Identified {error_df['is_high_error'].sum()} high-error samples out of {len(error_df)}")
        return error_df
    
    def analyze_error_patterns(self):
        """Find patterns in high error samples"""
        if self.error_df is None:
            raise ValueError("Must run compute_validation_errors first")
        
        # Compare feature distributions between high-error vs normal samples
        high_error_samples = self.error_df[self.error_df['is_high_error']]
        normal_samples = self.error_df[~self.error_df['is_high_error']]
        
        # Find features with significant differences
        feature_importance = {}
        for col in self.error_df.columns:
            if col in ['prediction_error', 'is_high_error', 'Target', 'Segment', 'Cluster']:
                continue
                
            # Simple measure: difference in mean values
            mean_diff = abs(high_error_samples[col].mean() - normal_samples[col].mean())
            # Normalize by standard deviation
            if self.error_df[col].std() > 0:
                mean_diff /= self.error_df[col].std()
            feature_importance[col] = mean_diff
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:5]  # Use top 5 features
        
        print("Top features associated with high error:")
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")
        
        self.high_error_features = [f[0] for f in top_features]
        return self.high_error_features
    
    def create_error_segment_rules(self):
        """Define rules for assigning samples to the error segment"""
        if self.high_error_features is None:
            self.analyze_error_patterns()
        
        rules = {}
        high_error_samples = self.error_df[self.error_df['is_high_error']]
        
        # For each key feature, determine a threshold
        for feature in self.high_error_features:
            # Simple approach: if feature is above its 75th percentile in high-error group
            threshold = high_error_samples[feature].quantile(0.75)
            rules[feature] = threshold
        
        self.error_segment_rules = rules
        return rules
    
    def identify_error_prone_samples(self, X):
        """Identify samples likely to have high prediction error"""
        if self.error_segment_rules is None:
            raise ValueError("Rules not created yet. Call create_error_segment_rules first.")
        
        # Start with all False
        is_error_prone = np.zeros(len(X), dtype=bool)
        
        # Apply rules: if several conditions are met, consider it error-prone
        matching_rules = np.zeros(len(X))
        
        for feature, threshold in self.error_segment_rules.items():
            if feature in X.columns:
                matching_rules += (X[feature] > threshold).astype(int)
        
        # If sample matches majority of rules, consider it error-prone
        rule_threshold = max(1, len(self.error_segment_rules) // 2)
        is_error_prone = matching_rules >= rule_threshold
        
        return is_error_prone

class EnhancedSegmenter(InsuranceSegmenter):
    """Extends the InsuranceSegmenter with an additional error-based segment"""
    
    def __init__(self, base_segments=4, error_analyzer=None):
        super().__init__()
        self.base_segments = base_segments
        self.error_analyzer = error_analyzer
        
    def create_enhanced_segments(self, X_train, y_train, X_val=None, y_val=None, y_val_pred=None):
        """Create the base segments plus an additional segment for error-prone samples"""
        # First create the regular segments
        base_segments = self.create_segments(X_train, y_train, method='kmeans', 
                                             n_clusters=self.base_segments)
        
        # If we don't have validation data with predictions, just return base segments
        if X_val is None or y_val is None or y_val_pred is None:
            print("Warning: No validation data provided. Using only base segments.")
            return base_segments
        
        # Analyze errors on validation data
        if self.error_analyzer is None:
            self.error_analyzer = ErrorAnalyzer()
            
        self.error_analyzer.compute_validation_errors(y_val, y_val_pred, X_val)
        self.error_analyzer.create_error_segment_rules()
        
        # Now enhance the segmentation
        X_train_copy = X_train.copy()
        
        # Mark error-prone samples in training data
        error_prone = self.error_analyzer.identify_error_prone_samples(X_train_copy)
        
        # Create a new segment ID (highest segment number + 1)
        error_segment_id = self.base_segments
        
        # Add the segment ID to the base segments
        enhanced_segments = base_segments.copy()
        enhanced_segments[error_prone] = error_segment_id
        
        print(f"Created error-based segment with {error_prone.sum()} samples")
        return enhanced_segments
    
    def assign_enhanced_segment(self, X):
        """Assign samples to segments including the error-prone segment"""
        # First assign to base segments
        segments = self.assign_segment(X)
        
        # Then check for error-prone
        if self.error_analyzer:
            error_prone = self.error_analyzer.identify_error_prone_samples(X)
            segments[error_prone] = self.base_segments  # Error segment ID
            
        return segments
