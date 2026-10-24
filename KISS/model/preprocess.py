import numpy as np
import pandas as pd

def engineer_features(df):
    """Engineer features with insurance domain knowledge for premium estimation"""
    # Health risk factors - critical for insurance pricing
    df['Health_Income_Ratio'] = df['HealthScore'] * df['AnnualIncome'] / 10000
    df['Health_Age_Risk'] = df['HealthScore'] / (df['Age'] + 1)
    df['Health_Credit_Score'] = df['HealthScore'] * df['CreditScore'] / 1000
    df['Health_Claims_Impact'] = df['HealthScore'] / (df['PreviousClaims'] + 1)
    
    # Polynomial features for HealthScore - common in actuarial models
    df['HealthScore_Squared'] = df['HealthScore'] ** 2
    df['HealthScore_Cubed'] = df['HealthScore'] ** 3
    
    # Insurance-specific risk factors
    df['Claims_Frequency'] = df['PreviousClaims'] / (df['InsuranceDuration'] + 1)
    df['Age_Risk_Factor'] = np.exp(df['Age'] / 50) - 1  # Exponential risk increase with age
    df['Income_Stability'] = df['AnnualIncome'] / (df['Age'] - 18 + 1)  # Income per working year
    
    # Insurance domain combinations
    df['Risk_Profile'] = df['HealthScore'] * (850 - df['CreditScore']) * df['PreviousClaims'] / 1000
    df['Coverage_Capacity'] = df['AnnualIncome'] / (df['NumberofDependents'] + 1) / 1000
    df['Financial_Stability'] = df['CreditScore'] * df['AnnualIncome'] / 100000
    
    # Geographic and demographic risk factors
    df['Rural_Health_Impact'] = df['HealthScore'] * (df['Location_Rural'].fillna(0) + 0.5)
    df['Urban_Health_Impact'] = df['HealthScore'] * (df['Location_Urban'].fillna(0) + 0.5)
    df['Education_Factor'] = df['EducationLevel_Bachelors'].fillna(0) + 2*df['EducationLevel_Masters'].fillna(0)
    df['Marital_Risk'] = df['MaritalStatus_Married'].fillna(0) * df['NumberofDependents']
    
    # Return engineered features and combinations to try
    combinations = [
        ('Health_Income_Ratio', 'Claims_Frequency'),
        ('Health_Age_Risk', 'Financial_Stability'),
        ('Health_Credit_Score', 'Risk_Profile'),
        ('Claims_Frequency', 'Coverage_Capacity'),
        ('Age_Risk_Factor', 'Rural_Health_Impact'),
        ('Income_Stability', 'Marital_Risk'),
        ('Risk_Profile', 'Education_Factor'),
        ('Coverage_Capacity', 'Urban_Health_Impact'),
        ('Financial_Stability', 'Health_Claims_Impact'),
        ('Rural_Health_Impact', 'Age_Risk_Factor'),
        ('Urban_Health_Impact', 'Income_Stability'),
        ('Education_Factor', 'Health_Credit_Score'),
        ('Marital_Risk', 'Financial_Stability'),
        ('HealthScore_Squared', 'Claims_Frequency'),
        ('HealthScore_Cubed', 'Coverage_Capacity'),
        ('Health_Claims_Impact', 'Risk_Profile')
    ]
    
    return df, combinations