import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from datetime import datetime
import os

class InsuranceDataProcessor:
    """
    Comprehensive data processor for insurance premium prediction
    """
    def __init__(self, train_path, test_path, results_dir='/home/USER/Projects/ikaggle/KISS/results'):
        self.train_path = train_path
        self.test_path = test_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize data containers
        self.train_raw = None
        self.test_raw = None
        self.train_processed = None
        self.test_processed = None
        self.target = None
        self.feature_info = {}
        
        # Define categorical and numerical features
        self.cat_features = ['Gender', 'MaritalStatus', 'EducationLevel', 'Occupation', 
                             'Location', 'PolicyType', 'CustomerFeedback', 'SmokingStatus', 
                             'ExerciseFrequency', 'PropertyType']
        
        self.num_features = ['Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore',
                             'PreviousClaims', 'VehicleAge', 'CreditScore', 'InsuranceDuration']
        
        self.date_features = ['PolicyStartDate']
        
        # Feature encoders
        self.encoders = {}
        self.scalers = {}
        
    def load_data(self):
        """Load raw data from files"""
        print("Loading raw data...")
        self.train_raw = pd.read_csv(self.train_path)
        self.test_raw = pd.read_csv(self.test_path)
        
        print(f"Train data shape: {self.train_raw.shape}")
        print(f"Test data shape: {self.test_raw.shape}")
        
        # Extract target variable if it exists in training data
        if 'PremiumAmount' in self.train_raw.columns:
            self.target = self.train_raw['PremiumAmount']
            self.train_raw = self.train_raw.drop('PremiumAmount', axis=1)
            print(f"Target variable extracted: {len(self.target)} values")
        else:
            print("Warning: Target variable 'PremiumAmount' not found in training data")
        
        return self
        
    def explore_data(self, save_plots=True):
        """Perform exploratory data analysis"""
        print("\nExploring data...")
        
        if self.train_raw is None:
            print("Data not loaded. Call load_data() first.")
            return self
            
        # Check missing values
        train_missing = self.train_raw.isnull().sum()
        test_missing = self.test_raw.isnull().sum()
        
        print("\nMissing values in training data:")
        print(train_missing[train_missing > 0])
        
        print("\nMissing values in test data:")
        print(test_missing[test_missing > 0])
        
        # Check data types
        print("\nData types:")
        print(self.train_raw.dtypes)
        
        # Check basic statistics
        print("\nBasic statistics for numerical features:")
        print(self.train_raw[self.num_features].describe())
        
        # Check categorical distributions
        for cat in self.cat_features:
            if (cat in self.train_raw.columns):
                value_counts = self.train_raw[cat].value_counts(dropna=False)
                self.feature_info[cat] = {
                    'unique_values': len(value_counts),
                    'top_values': value_counts.head(5).to_dict()
                }
                
                # Create distribution plots for categorical features
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    sns.countplot(y=cat, data=self.train_raw, order=value_counts.index[:10])
                    plt.title(f'Distribution of {cat}')
                    plt.tight_layout()
                    plt.savefig(f'{self.results_dir}/cat_dist_{cat}.png')
                    plt.close()
        
        # Create distribution plots for numerical features
        for num in self.num_features:
            if num in self.train_raw.columns:
                self.feature_info[num] = {
                    'mean': self.train_raw[num].mean(),
                    'median': self.train_raw[num].median(),
                    'std': self.train_raw[num].std(),
                    'min': self.train_raw[num].min(),
                    'max': self.train_raw[num].max()
                }
                
                if save_plots:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(self.train_raw[num].dropna(), kde=True)
                    plt.title(f'Distribution of {num}')
                    plt.tight_layout()
                    plt.savefig(f'{self.results_dir}/num_dist_{num}.png')
                    plt.close()
        
        # Check target distribution if available
        if self.target is not None and save_plots:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.target, kde=True)
            plt.title('Distribution of Premium Amount')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/target_distribution.png')
            plt.close()
            
            # Log transform of target
            plt.figure(figsize=(10, 6))
            sns.histplot(np.log1p(self.target), kde=True)
            plt.title('Distribution of Log(Premium Amount)')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/target_log_distribution.png')
            plt.close()
        
        return self
        
    def clean_data(self):
        """Clean the data by handling missing values and outliers"""
        print("\nCleaning data...")
        
        # Create copies to avoid modifying originals
        self.train_processed = self.train_raw.copy()
        self.test_processed = self.test_raw.copy()
        
        # Handle missing values for each feature type
        for cat in self.cat_features:
            if cat in self.train_processed.columns:
                # For categorical, use a 'missing' category
                self.train_processed[cat] = self.train_processed[cat].fillna('missing')
                self.test_processed[cat] = self.test_processed[cat].fillna('missing')
        
        # For numerical features, use domain-appropriate imputation
        for num in self.num_features:
            if num in self.train_processed.columns:
                # For income and credit scores, impute with median due to skewness
                if num in ['AnnualIncome', 'CreditScore']:
                    median_value = self.train_processed[num].median()
                    self.train_processed[num] = self.train_processed[num].fillna(median_value)
                    self.test_processed[num] = self.test_processed[num].fillna(median_value)
                # For health scores, impute with mean (assuming normal distribution)
                elif num == 'HealthScore':
                    mean_value = self.train_processed[num].mean()
                    self.train_processed[num] = self.train_processed[num].fillna(mean_value)
                    self.test_processed[num] = self.test_processed[num].fillna(mean_value)
                # For count variables, impute with 0
                elif num in ['NumberofDependents', 'PreviousClaims']:
                    self.train_processed[num] = self.train_processed[num].fillna(0)
                    self.test_processed[num] = self.test_processed[num].fillna(0)
                # For other numericals, use median
                else:
                    median_value = self.train_processed[num].median()
                    self.train_processed[num] = self.train_processed[num].fillna(median_value)
                    self.test_processed[num] = self.test_processed[num].fillna(median_value)
        
        # Handle date features
        for date_col in self.date_features:
            if date_col in self.train_processed.columns:
                # Extract useful components from dates
                for df in [self.train_processed, self.test_processed]:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df[f'{date_col}_Year'] = df[date_col].dt.year
                        df[f'{date_col}_Month'] = df[date_col].dt.month
                        df[f'{date_col}_Day'] = df[date_col].dt.day
                        df[f'{date_col}_DayOfWeek'] = df[date_col].dt.dayofweek
                        
                        # Drop original date column
                        df.drop(date_col, axis=1, inplace=True)
                    except:
                        print(f"Warning: Could not process date column {date_col}")
        
        # Handle outliers in numerical features
        for num in self.num_features:
            if num in self.train_processed.columns:
                # Get percentiles
                q1 = self.train_processed[num].quantile(0.01)
                q3 = self.train_processed[num].quantile(0.99)
                iqr = q3 - q1
                
                # Set boundaries for outliers
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Cap outliers
                self.train_processed[num] = np.clip(
                    self.train_processed[num], 
                    lower_bound, 
                    upper_bound
                )
                self.test_processed[num] = np.clip(
                    self.test_processed[num], 
                    lower_bound, 
                    upper_bound
                )
        
        print(f"Data cleaning complete. Train shape: {self.train_processed.shape}, Test shape: {self.test_processed.shape}")
        return self
    
    def engineer_features(self):
        """Engineer features to improve model performance"""
        print("\nEngineering features...")
        
        # Create interaction features between important variables
        interactions = [
            ('Age', 'HealthScore'),  # Age-health interaction
            ('AnnualIncome', 'CreditScore'),  # Financial stability
            ('AnnualIncome', 'NumberofDependents'),  # Financial burden
            ('HealthScore', 'SmokingStatus'),  # Health risk
            ('Age', 'PreviousClaims'),  # Risk history
            ('InsuranceDuration', 'PreviousClaims')  # Loyalty vs risk
        ]
        
        # Create ratio features
        for df in [self.train_processed, self.test_processed]:
            # Interaction terms
            for feat1, feat2 in interactions:
                if feat1 in df.columns and feat2 in df.columns:
                    if feat2 == 'SmokingStatus':
                        # Convert Yes/No to 1/0
                        smoking_binary = (df[feat2] == 'Yes').astype(int)
                        df[f'{feat1}_{feat2}_Interaction'] = df[feat1] * smoking_binary
                    else:
                        df[f'{feat1}_{feat2}_Interaction'] = df[feat1] * df[feat2]
            
            # Create important ratios
            if 'AnnualIncome' in df.columns and 'NumberofDependents' in df.columns:
                df['IncomePerDependent'] = df['AnnualIncome'] / (df['NumberofDependents'] + 1)
                
            if 'Age' in df.columns and 'HealthScore' in df.columns:
                df['HealthScoreToAgeRatio'] = df['HealthScore'] / df['Age']
                
            if 'PreviousClaims' in df.columns and 'InsuranceDuration' in df.columns:
                df['ClaimsPerYear'] = df['PreviousClaims'] / (df['InsuranceDuration'] + 0.1)
                
            # Insurance risk score (composite feature)
            risk_features = ['Age', 'HealthScore', 'PreviousClaims', 'InsuranceDuration']
            if all(feat in df.columns for feat in risk_features):
                # Higher age, lower health, more claims = higher risk
                df['InsuranceRiskScore'] = (
                    df['Age'] / 50 +  # Age factor
                    (10 - df['HealthScore'])/10 +  # Inverse health factor
                    df['PreviousClaims'] / (df['InsuranceDuration'] + 1)  # Claims rate
                )
        
        return self
    
    def encode_features(self):
        """Encode categorical features"""
        print("\nEncoding categorical features...")
        
        # For each categorical feature, use appropriate encoding
        for cat in self.cat_features:
            if cat in self.train_processed.columns:
                # For binary categoricals, use simple mapping
                if self.feature_info.get(cat, {}).get('unique_values', 0) <= 2 or cat in ['SmokingStatus']:
                    unique_vals = self.train_processed[cat].unique()
                    
                    # Binary encoding for yes/no features - ensure they map to exactly 0 or 1
                    if set(unique_vals) <= {'Yes', 'No', 'missing'} or cat == 'SmokingStatus':
                        print(f"Binary encoding {cat} as 0-1")
                        self.train_processed[cat] = self.train_processed[cat].map({'Yes': 1, 'No': 0, 'missing': 0})
                        self.test_processed[cat] = self.test_processed[cat].map({'Yes': 1, 'No': 0, 'missing': 0})
                    else:
                        # For other binary features, use one-hot encoding
                        # FIX: Use sparse_output instead of sparse
                        try:
                            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        except TypeError:
                            # For backward compatibility with older scikit-learn versions
                            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        
                        train_encoded = encoder.fit_transform(self.train_processed[[cat]])
                        test_encoded = encoder.transform(self.test_processed[[cat]])
                        
                        # Create new column names
                        encoded_cols = [f"{cat}_{val}" for val in encoder.categories_[0]]
                        
                        # Add encoded columns to dataframes
                        for i, col in enumerate(encoded_cols):
                            self.train_processed[col] = train_encoded[:, i]
                            self.test_processed[col] = test_encoded[:, i]
                        
                        # Drop original column
                        self.train_processed.drop(cat, axis=1, inplace=True)
                        self.test_processed.drop(cat, axis=1, inplace=True)
                
                # For categorical features with multiple levels, use one-hot encoding
                else:
                    # FIX: Use sparse_output instead of sparse
                    try:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    except TypeError:
                        # For backward compatibility with older scikit-learn versions
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    
                    train_encoded = encoder.fit_transform(self.train_processed[[cat]])
                    test_encoded = encoder.transform(self.test_processed[[cat]])
                    
                    # Store encoder
                    self.encoders[cat] = encoder
                    
                    # Create new column names
                    encoded_cols = [f"{cat}_{val}" for val in encoder.categories_[0]]
                    
                    # Add encoded columns to dataframes
                    for i, col in enumerate(encoded_cols):
                        self.train_processed[col] = train_encoded[:, i]
                        self.test_processed[col] = test_encoded[:, i]
                    
                    # Drop original column
                    self.train_processed.drop(cat, axis=1, inplace=True)
                    self.test_processed.drop(cat, axis=1, inplace=True)
        
        # Make sure binary features remain between 0-1 (extra safeguard)
        binary_features = ['SmokingStatus', 'Gender']
        for feat in binary_features:
            if feat in self.train_processed.columns:
                self.train_processed[feat] = self.train_processed[feat].clip(0, 1)
                self.test_processed[feat] = self.test_processed[feat].clip(0, 1)
        
        return self
    
    def scale_features(self):
        """Scale numerical features for better model performance"""
        print("\nScaling numerical features...")
        
        # Get all numerical columns after feature engineering
        num_cols = self.train_processed.select_dtypes(include=np.number).columns.tolist()
        
        # Create a list of binary features & one-hot encoded features that should never be scaled
        binary_features = ['SmokingStatus', 'Gender'] 
        
        # Identify one-hot encoded features (they should not be scaled)
        one_hot_features = []
        for col in num_cols:
            if '_' in col and col.split('_')[0] in self.cat_features:
                one_hot_features.append(col)
                print(f"Identified {col} as one-hot feature (will not be scaled)")
        
        # Remove binary features & one-hot features from scaling
        for feat in binary_features + one_hot_features:
            if feat in num_cols:
                print(f"Removing {feat} from scaling as it should remain binary/categorical")
                num_cols.remove(feat)
        
        # List of features that should never be negative
        non_negative_features = [
            'Age', 'AnnualIncome', 'NumberofDependents', 'HealthScore', 
            'PreviousClaims', 'VehicleAge', 'CreditScore', 'InsuranceDuration',
            'id'  # IDs should never be negative
        ]
        
        # Identify features that should use MinMaxScaler
        date_components = [col for col in num_cols if 'Date' in col]
        interaction_features = [col for col in num_cols if 'Interaction' in col]
        ratio_features = [col for col in num_cols if any(term in col for term in 
                                                    ['Ratio', 'Per', 'PerDependent', 'PerYear'])]
        
        # Also add any basic features that should never be negative
        natural_positive_features = [col for col in num_cols if col in non_negative_features]
        
        # Combine all features that should use MinMaxScaler
        minmax_features = date_components + interaction_features + ratio_features + natural_positive_features
        
        print("Using MinMaxScaler for the following features to avoid negative values:")
        print(", ".join(minmax_features))
        
        # Use different scalers based on feature type
        for col in num_cols:
            if col in minmax_features:
                # Use MinMaxScaler (0-1 scaling) for features that should never be negative
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(0, 1))
            else:
                # Use StandardScaler for other numerical features
                scaler = StandardScaler()
                
            # Apply the appropriate scaler
            self.train_processed[col] = scaler.fit_transform(self.train_processed[[col]])
            self.test_processed[col] = scaler.transform(self.test_processed[[col]])
            self.scalers[col] = scaler
        
        # Ensure binary and one-hot features stay between 0 and 1
        for feat in binary_features + one_hot_features:
            if feat in self.train_processed.columns:
                self.train_processed[feat] = self.train_processed[feat].clip(0, 1)
                self.test_processed[feat] = self.test_processed[feat].clip(0, 1)
                print(f"Clipped {feat} to 0-1 range after processing")
        
        return self
    
    def prepare_data(self):
        """Run the full data preparation pipeline"""
        return (
            self.load_data()
                .explore_data()
                .clean_data()
                .engineer_features()
                .encode_features()
                .scale_features()
        )
    
    def get_processed_data(self):
        """Return processed data"""
        if self.train_processed is None:
            print("Data not processed. Call prepare_data() first.")
            return None, None, None
            
        X_train = self.train_processed
        X_test = self.test_processed
        y_train = self.target
        
        # Log final data shapes
        print("\nFinal data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        if y_train is not None:
            print(f"y_train: {y_train.shape}")
        
        return X_train, X_test, y_train
    
    def create_feature_subsets(self, n_subsets=5):
        """Create feature subsets for diverse models"""
        if self.train_processed is None:
            print("Data not processed. Call prepare_data() first.")
            return []
            
        features = list(self.train_processed.columns)
        
        # Calculate feature importance using a simple correlation with target
        if self.target is not None:
            importance = {}
            for col in features:
                corr = np.abs(np.corrcoef(
                    self.train_processed[col], 
                    np.log1p(self.target)
                )[0, 1])
                importance[col] = corr
                
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            sorted_features = [f[0] for f in sorted_features]
        else:
            sorted_features = features
        
        # Create diverse feature subsets
        feature_subsets = []
        
        # Subset 1: All features
        feature_subsets.append(sorted_features)
        
        # Subset 2: Top 30% features
        top_k = int(0.3 * len(sorted_features))
        feature_subsets.append(sorted_features[:top_k])
        
        # Subset 3: Only features with names containing health/medical terms
        health_features = [f for f in sorted_features if 
                          any(term in f.lower() for term in ['health', 'smoke', 'exercise'])]
        feature_subsets.append(health_features)
        
        # Subset 4: Only features with names containing financial terms
        financial_features = [f for f in sorted_features if 
                             any(term in f.lower() for term in ['income', 'credit', 'premium'])]
        feature_subsets.append(financial_features)
        
        # Subset 5: Features with demographic information
        demographic_features = [f for f in sorted_features if 
                               any(term in f.lower() for term in ['age', 'gender', 'marital', 'dependent'])]
        feature_subsets.append(demographic_features)
        
        # Ensure we have enough features in each subset
        for i in range(len(feature_subsets)):
            if len(feature_subsets[i]) < 5:
                feature_subsets[i].extend(sorted_features[:5])
        
        return feature_subsets
