import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers, lower_bound, upper_bound

def analyze_features():
    # Load data
    train_df = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'train.csv')
    
    # Basic information
    print("\nDataset Info:")
    print(train_df.info())
    
    print("\nNumerical Features Summary:")
    print(train_df.describe())
    
    print("\nMissing Values:")
    print(train_df.isnull().sum())
    
    print("\nFeature Types:")
    for col in train_df.columns:
        print(f"{col}: {train_df[col].nunique()} unique values")
        if train_df[col].dtype == 'object':
            print(train_df[col].value_counts(), "\n")
    
    # Outlier detection for numerical features
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    print("\nOutlier Analysis:")
    for col in numerical_cols:
        outliers, lower, upper = detect_outliers(train_df, col)
        pct_outliers = (len(outliers) / len(train_df)) * 100
        print(f"\n{col}:")
        print(f"Number of outliers: {len(outliers)}")
        print(f"Percentage of outliers: {pct_outliers:.2f}%")
        print(f"Outlier boundaries: [{lower:.2f}, {upper:.2f}]")
        
        # Box plot with outliers
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=train_df[col])
        plt.title(f'Distribution of {col} with Outliers')
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'notebooks' / f'outliers_{col}.png')
        
        if col == 'PremiumAmount':
            # Special focus on target variable outliers
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            sns.histplot(train_df[col], bins=50)
            plt.title('Premium Distribution (with outliers)')
            
            plt.subplot(1, 2, 2)
            cleaned_premiums = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)][col]
            sns.histplot(cleaned_premiums, bins=50)
            plt.title('Premium Distribution (without outliers)')
            plt.tight_layout()
            plt.savefig(PROJECT_ROOT / 'notebooks' / 'premium_distribution_comparison.png')
    
    # Correlation analysis for numerical features
    plt.figure(figsize=(10, 8))
    numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    sns.heatmap(train_df[numerical_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'notebooks' / 'correlation_matrix.png')
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(train_df['PremiumAmount'], bins=50)
    plt.title('Distribution of Premium Amount')
    plt.savefig(PROJECT_ROOT / 'notebooks' / 'premium_distribution.png')
    
    # Boxplots for categorical variables
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=col, y='PremiumAmount', data=train_df)
        plt.xticks(rotation=45)
        plt.title(f'Premium Amount by {col}')
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'notebooks' / f'premium_by_{col}.png')

if __name__ == "__main__":
    analyze_features()
