import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
from pathlib import Path

class FeatureSelector:
    def __init__(self, n_features: int = 50):
        self.n_features = n_features
        self.selected_features = None
        self.importance_scores = None
        
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Calculate feature importance using multiple methods"""
        importance_scores = pd.DataFrame()
        
        # 1. Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
        rf.fit(X, y)
        importance_scores['rf_score'] = rf.feature_importances_
        
        # 2. LightGBM importance
        lgb_model = lgb.LGBMRegressor(n_estimators=100)
        lgb_model.fit(X, y)
        importance_scores['lgb_score'] = lgb_model.feature_importances_
        
        # 3. Mutual Information
        mi_scores = mutual_info_regression(StandardScaler().fit_transform(X), y)
        importance_scores['mi_score'] = mi_scores
        
        # Calculate aggregate score
        importance_scores['aggregate_score'] = (
            importance_scores['rf_score'] * 0.4 +
            importance_scores['lgb_score'] * 0.4 +
            importance_scores['mi_score'] * 0.2
        )
        
        importance_scores.index = X.columns
        return importance_scores.sort_values('aggregate_score', ascending=False)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select top features based on importance scores"""
        self.importance_scores = self.get_feature_importance(X, y)
        self.selected_features = list(self.importance_scores.index[:self.n_features])
        return self.selected_features
    
    def plot_importance(self, top_n: int = 20):
        """Plot feature importance scores"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 6))
        scores = self.importance_scores.head(top_n)
        sns.barplot(x=scores['aggregate_score'], y=scores.index)
        plt.title(f'Top {top_n} Feature Importance Scores')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        return plt

def main():
    # Example usage
    data_dir = Path('/home/USER/Projects/ikaggle/data/processed/cleaned_train.csv')
    train_data = pd.read_csv(data_dir)
    
    X = train_data.drop(['id', 'PremiumAmount'], axis=1)
    y = train_data['PremiumAmount']
    
    # Select top features
    selector = FeatureSelector(n_features=50)
    selected_features = selector.select_features(X, y)
    
    # Plot importance scores
    import matplotlib.pyplot as plt
    selector.plot_importance(top_n=20)
    plt.show()
    
    print("Selected features:", selected_features)

if __name__ == "__main__":
    main()
