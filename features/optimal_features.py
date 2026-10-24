import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class OptimalFeatureSelector:
    def __init__(self):
        # Features selected based on combined analysis
        self.primary_features = [
            'AnnualIncome',
            'Age',
            'CreditScore',
            'HealthScore',
            'NumberofDependents'
        ]
        
        # Features that showed strong interaction effects
        self.interaction_pairs = [
            ('Age', 'HealthScore'),
            ('AnnualIncome', 'CreditScore'),
            ('HealthScore', 'NumberofDependents')
        ]

    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for numerical columns"""
        # Create squared terms for important numerical features
        for feature in self.primary_features:
            df[f'{feature}_Squared'] = df[feature] ** 2
        
        # Create interaction terms
        for feat1, feat2 in self.interaction_pairs:
            df[f'{feat1}_{feat2}_Interaction'] = df[feat1] * df[feat2]
        
        return df

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful ratio features"""
        # Income per dependent
        df['IncomePerDependent'] = df['AnnualIncome'] / (df['NumberofDependents'] + 1)
        
        # Health-Age Index
        df['HealthAgeIndex'] = df['HealthScore'] / df['Age']
        
        # Credit-Income Ratio
        df['CreditIncomeRatio'] = df['CreditScore'] / df['AnnualIncome']
        
        return df

    def create_bucketed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bucketed versions of continuous features"""
        df['AgeGroup'] = pd.qcut(df['Age'], q=5, labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'Elderly'])
        df['IncomeLevel'] = pd.qcut(df['AnnualIncome'], q=5, labels=['Low', 'ModLow', 'Moderate', 'ModHigh', 'High'])
        df['HealthLevel'] = pd.qcut(df['HealthScore'], q=5, labels=['Poor', 'Fair', 'Good', 'VeryGood', 'Excellent'])
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transformations"""
        df = df.copy()
        
        # Create all derived features
        df = self.create_polynomial_features(df)
        df = self.create_ratio_features(df)
        df = self.create_bucketed_features(df)
        
        # Create dummy variables for categorical features
        categorical_columns = ['AgeGroup', 'IncomeLevel', 'HealthLevel']
        df = pd.get_dummies(df, columns=categorical_columns, prefix_sep='_')
        
        return df

    def get_feature_list(self) -> list:
        """Return list of all features after transformation"""
        base_features = self.primary_features.copy()
        
        # Add polynomial features
        squared_features = [f"{f}_Squared" for f in self.primary_features]
        
        # Add interaction features
        interaction_features = [f"{f1}_{f2}_Interaction" for f1, f2 in self.interaction_pairs]
        
        # Add ratio features
        ratio_features = [
            'IncomePerDependent',
            'HealthAgeIndex',
            'CreditIncomeRatio'
        ]
        
        # Add categorical features (after dummy encoding)
        categorical_features = [
            'AgeGroup', 'IncomeLevel', 'HealthLevel'
        ]
        
        return {
            'base': base_features,
            'squared': squared_features,
            'interactions': interaction_features,
            'ratios': ratio_features,
            'categorical': categorical_features
        }

