import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple

class SubmissionFeatureSelector:
    def __init__(self, importance_threshold: float = 0.01):
        self.importance_threshold = importance_threshold
        self.feature_importances = None
        self.selected_features = None
        self.feature_groups = None
        
    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[Tuple[str, float]]]:
        """Analyze and group features by their importance"""
        # Get base importance scores
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Create importance DataFrame
        self.feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_,
            'type': [self._get_feature_type(col) for col in X.columns]
        }).sort_values('importance', ascending=False)
        
        # Group features by type and importance
        self.feature_groups = {
            'critical': [],    # importance > 0.1
            'important': [],   # 0.05 < importance <= 0.1
            'supporting': [],  # 0.01 < importance <= 0.05
            'noise': []       # importance <= 0.01
        }
        
        for _, row in self.feature_importances.iterrows():
            imp = row['importance']
            feature = (row['feature'], imp)
            
            if imp > 0.1:
                self.feature_groups['critical'].append(feature)
            elif imp > 0.05:
                self.feature_groups['important'].append(feature)
            elif imp > 0.01:
                self.feature_groups['supporting'].append(feature)
            else:
                self.feature_groups['noise'].append(feature)
        
        # Select features above threshold
        self.selected_features = self.feature_importances[
            self.feature_importances['importance'] > self.importance_threshold
        ]['feature'].tolist()
        
        return self.feature_groups
    
    def _get_feature_type(self, column: str) -> str:
        """Determine feature type based on name"""
        if '_Interaction' in column:
            return 'interaction'
        elif '_Squared' in column:
            return 'polynomial'
        elif any(x in column for x in ['Ratio', 'Index', 'Per']):
            return 'ratio'
        elif any(x in column for x in ['Group', 'Level', 'Type']):
            return 'categorical'
        return 'base'
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get detailed summary of feature importance by type"""
        if self.feature_importances is None:
            raise ValueError("Must run analyze_features first")
        
        summary = self.feature_importances.groupby('type').agg({
            'importance': ['count', 'mean', 'sum'],
            'feature': lambda x: ', '.join(x[:3] + ['...'] if len(x) > 3 else x)
        })
        
        summary.columns = ['count', 'mean_importance', 'total_importance', 'example_features']
        return summary.sort_values('total_importance', ascending=False)
    
    def print_feature_report(self):
        """Print detailed feature importance report"""
        print("Feature Importance Report")
        print("=" * 50)
        
        for group, features in self.feature_groups.items():
            if features:
                print(f"\n{group.title()} Features:")
                print("-" * 30)
                for feature, importance in features:
                    print(f"{feature:<30} {importance:.4f}")
        
        print("\nFeature Type Summary:")
        print("-" * 30)
        print(self.get_feature_summary().to_string())
    
    def transform_for_submission(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features for final submission using only important features"""
        if self.selected_features is None:
            raise ValueError("Must run analyze_features first")
            
        return X[self.selected_features].copy()

