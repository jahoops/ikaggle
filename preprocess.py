import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
import logging

# Ensure data/raw directory exists
if not os.path.exists('data/raw'):
    raise FileNotFoundError("'data/raw' directory not found. Please create it and place train.csv and test.csv inside.")

for file in ['train.csv', 'test.csv']:
    if not os.path.exists(f'data/raw/{file}'):
        raise FileNotFoundError(f"Required file 'data/raw/{file}' not found.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

numeric_columns = [
    'Age', 'AnnualIncome', 'PremiumAmount', 'InsuranceDuration', 'PreviousClaims',
    'NumberofDependents', 'HealthScore', 'VehicleAge', 'CreditScore' 
]

onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def remove_high_corr_features(df, threshold=0.95):
    """Remove features with pairwise correlation above threshold."""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    logging.info(f"Dropping {len(to_drop)} highly correlated features")
    df = df.drop(columns=to_drop, errors='ignore')
    return df

def remove_low_variance_features(df, threshold=0.01):
    """Remove features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    to_drop = df.columns[~selector.get_support()]
    logging.info(f"Dropping {len(to_drop)} low-variance features: {list(to_drop)}")
    df = df.drop(columns=to_drop, errors='ignore')
    return df

def iqr_outlier_clip(series, lower_quantile=0.25, upper_quantile=0.75, factor=1.5):
    """Clip outliers using IQR rule."""
    q1 = series.quantile(lower_quantile)
    q3 = series.quantile(upper_quantile)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return np.clip(series, lower_bound, upper_bound)

def handle_policy_start_date(df):
    """Extract year and month using simple string manipulation."""
    date_column = None
    if 'PolicyStartDate' in df.columns:
        date_column = 'PolicyStartDate'
    elif 'Policy Start Date' in df.columns:
        date_column = 'Policy Start Date'
    else:
        logger.error(f"Available columns: {df.columns.tolist()}")
        raise KeyError("Required date column not found in the dataset.")
    
    def extract_year(date_str):
        try:
            # Take first 4 chars for year
            year_str = str(date_str)[:4]
            year = int(year_str)
            # Basic validation (adjust range as needed)
            if 1900 <= year <= 2100:
                return year
            return 2000  # Default year if invalid
        except (ValueError, IndexError):
            return 2000  # Default year if parsing fails
    
    def extract_month(date_str):
        try:
            # Skip first 4 chars (year) and take next 2 (month)
            month_str = str(date_str)[4:6]
            month = int(month_str)
            # Validate month is between 1 and 12
            if 1 <= month <= 12:
                return month
            return 0
        except (ValueError, IndexError):
            return 0
    
    # Extract year and month
    df['SignupYear'] = df[date_column].apply(extract_year)
    df['SignupMonth'] = df[date_column].apply(extract_month)
    
    # Log distributions
    logger.info(f"Year distribution:\n{df['SignupYear'].value_counts().sort_index()}")
    logger.info(f"Month distribution:\n{df['SignupMonth'].value_counts().sort_index()}")
    
    # Drop original date column
    df = df.drop(date_column, axis=1)
    
    # Add SignupYear to numeric columns if not already present
    if 'SignupYear' not in numeric_columns:
        numeric_columns.append('SignupYear')
        logger.info("Added 'SignupYear' to numeric_columns")
    
    return df

def clean_categorical_values(df, categorical_columns):
    """Clean categorical feature values by removing special characters."""
    for col in categorical_columns:
        df[col] = df[col].astype(str).str.replace('[^A-Za-z0-9_]+', '', regex=True)
    return df

def get_feature_names(encoder, categorical_columns):
    """Get feature names for one-hot encoded columns"""
    feature_names = []
    for i, column in enumerate(categorical_columns):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{column}_{category}")
    return feature_names

def process_dataset(df, is_train=True, train_medians=None, train_modes=None):
    """Process dataset with consistent steps for both train and test."""
    logger.info(f"Processing {'training' if is_train else 'test'} dataset...")
    
    # Clean column names
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)
    
    # Determine categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    logger.info(f"Identified categorical columns: {categorical_columns}")
    
    # Clean categorical values
    df = clean_categorical_values(df, categorical_columns)
    
    # Process dates
    df = handle_policy_start_date(df)
    
    # Update categorical columns after date processing
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Fill missing values
    for col in numeric_columns:
        if col == "PremiumAmount" and not is_train:
            continue
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            if is_train:
                if col == 'SignupMonth':
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
                else:
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
            else:
                # Use training data statistics for test data
                if col == 'SignupMonth':
                    df[col] = df[col].fillna(train_modes[col])
                else:
                    df[col] = df[col].fillna(train_medians[col])
    
    # Fill categorical missing values
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    # Handle outliers
    for col in numeric_columns:
        if col == "PremiumAmount" and not is_train:
            continue
        if df[col].min() < -1e6 or df[col].max() > 1e6:
            df[col] = df[col].clip(lower=-1e6, upper=1e6)
    
    # One-Hot Encoding
    if is_train:
        categorical_data = pd.DataFrame(
            onehot_encoder.fit_transform(df[categorical_columns]),
            columns=get_feature_names(onehot_encoder, categorical_columns)
        )
    else:
        categorical_data = pd.DataFrame(
            onehot_encoder.transform(df[categorical_columns]),
            columns=get_feature_names(onehot_encoder, categorical_columns)
        )
    
    df = df.drop(columns=categorical_columns).reset_index(drop=True)
    df = pd.concat([df, categorical_data], axis=1)
    
    return df

try:
    # Process training data
    train_data = pd.read_csv('data/raw/train.csv')
    train_data = process_dataset(train_data, is_train=True)
    
    # Split into train and validate before feature selection
    train_data, validate_data = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42
    )
    
    # Save training statistics for test data
    train_medians = {col: train_data[col].median() for col in numeric_columns if col != 'PremiumAmount'}
    train_modes = {col: train_data[col].mode()[0] for col in numeric_columns if col != 'PremiumAmount'}
    
    # Process test data using training statistics
    test_data = pd.read_csv('data/raw/test.csv')
    test_data = process_dataset(test_data, is_train=False, 
                              train_medians=train_medians, 
                              train_modes=train_modes)
    
    # Remove low-variance and highly correlated features from datasets
    train_data = remove_low_variance_features(train_data)
    train_data = remove_high_corr_features(train_data, threshold=0.95)
    
    # Select features
    selected_features = list(train_data.columns)
    # Remove 'id' and 'PremiumAmount' from selected features
    for col in ['id', 'PremiumAmount']:
        if col in selected_features:
            selected_features.remove(col)
    
    # Prepare final datasets with selected features
    train_features = train_data[selected_features]
    validate_features = validate_data[selected_features]
    test_features = test_data[selected_features]
    
    # Reconstruct final datasets
    train_data = pd.concat([
        train_features,
        train_data[['PremiumAmount']],
        train_data[['id']]
    ], axis=1)
    
    validate_data = pd.concat([
        validate_features,
        validate_data[['PremiumAmount']], 
        validate_data[['id']]
    ], axis=1)
    
    test_data = pd.concat([
        test_features,
        pd.Series([0] * len(test_features), name='PremiumAmount'),
        test_data[['id']]
    ], axis=1)
    
    logger.info("Re-appended 'PremiumAmount' and 'id' after scaling")
    
    # Save processed datasets and artifacts
    os.makedirs('data/processed', exist_ok=True)
    train_data.to_csv('data/processed/cleaned_train.csv', index=False)
    validate_data.to_csv('data/processed/cleaned_validate.csv', index=False)
    test_data.to_csv('data/processed/cleaned_test.csv', index=False)
    
    logger.info("Preprocessing completed successfully")
    logger.info(f"Training shape: {train_data.shape}")
    logger.info(f"Validation shape: {validate_data.shape}")
    logger.info(f"Test shape: {test_data.shape}")

except Exception as e:
    logger.error(f"Error during preprocessing: {str(e)}")
    raise
