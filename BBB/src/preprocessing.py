import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_val_score
from datetime import datetime
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
# ...existing imports...

def create_interactions(df):
    """Create interaction features"""
    df['Income_per_Dependent'] = df['AnnualIncome'] / (df['NumberofDependents'] + 1)
    df['Health_Age_Ratio'] = df['HealthScore'] / df['Age']
    df['Claims_per_Duration'] = df['PreviousClaims'] / (df['InsuranceDuration'] + 1)
    df['Income_Credit_Interaction'] = df['AnnualIncome'] * df['CreditScore']
    return df

def create_bins(df):
    """Create binned categories from numerical features"""
    df['AgeGroup'] = pd.qcut(df['Age'], q=5, labels=['VeryYoung', 'Young', 'Middle', 'Senior', 'Elderly'])
    df['IncomeLevel'] = pd.qcut(df['AnnualIncome'], q=5, labels=['Low', 'LowMid', 'Mid', 'MidHigh', 'High'])
    df['HealthGroup'] = pd.qcut(df['HealthScore'], q=4, labels=['Poor', 'Fair', 'Good', 'Excellent'])
    return df

def extract_date_features(df):
    """Extract detailed date features"""
    df['PolicyAge'] = (datetime.now() - df['PolicyStartDate']).dt.days / 365
    df['PolicyQuarter'] = df['PolicyStartDate'].dt.quarter
    df['IsYearEnd'] = df['PolicyStartDate'].dt.month.isin([11, 12, 1]).astype(int)
    return df

def create_domain_features(df):
    """Create insurance domain-specific features"""
    df['RiskScore'] = (df['Age'] * 0.3 + 
                       df['HealthScore'] * -0.4 + 
                       df['PreviousClaims'] * 0.3)
    df['IsHighRisk'] = ((df['Age'] > 60) & 
                        (df['HealthScore'] < df['HealthScore'].median()) & 
                        (df['PreviousClaims'] > 0)).astype(int)
    df['HasDependents'] = (df['NumberofDependents'] > 0).astype(int)
    return df

def create_advanced_features_numeric(df):
    """Create advanced features for already encoded data"""
    # Only create numeric interactions, no string operations
    if 'Age' in df.columns:
        df['Age_Squared'] = df['Age'] ** 2
    if 'AnnualIncome' in df.columns:
        df['Income_Squared'] = df['AnnualIncome'] ** 2
    if all(col in df.columns for col in ['Age', 'HealthScore', 'AnnualIncome']):
        df['Age_Health_Income'] = df['Age'] * df['HealthScore'] * df['AnnualIncome']
    
    return df

def encode_categorical_columns(df):
    """Encode categorical columns using LabelEncoder"""
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoded_df = df.copy()
    encoders = {}
    
    # First impute missing values in categorical columns
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    
    for column in categorical_columns:
        # Impute missing values first - fix the dimensionality issue
        imputed_values = cat_imputer.fit_transform(df[[column]]).ravel()  # Convert to 1D array
        encoded_df[column] = imputed_values
        
        # Then encode
        encoders[column] = LabelEncoder()
        encoded_df[column] = encoders[column].fit_transform(encoded_df[column])
    
    return encoded_df, encoders

def create_test_data(df, sample_size=1000):
    """Create a smaller dataset while preserving index"""
    if isinstance(df, pd.DataFrame):
        # Sample while preserving index
        sampled = df.sample(n=min(sample_size, len(df)), random_state=42)
        # Reset index to avoid indexing issues
        return sampled.reset_index(drop=True)
    return df

def preprocess_data(train_df, test_df, config):
    """Preprocess training and test data"""
    try:
        # Reset indices for both dataframes
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        print("\nChecking for missing values:")
        print(train_df.isnull().sum())
        
        # Try to identify the target column
        potential_target_cols = [
            col for col in train_df.columns if 
            any(keyword in col.lower() for keyword in 
                ['premium', 'target', 'amount', 'price', 'cost'])
        ]
        
        print("\nPotential target columns found:", potential_target_cols)
        
        if not potential_target_cols:
            raise ValueError("No potential target column found. Please check the data.")
        
        target_col = potential_target_cols[0]
        print(f"\nUsing '{target_col}' as target column")
        
        # Extract target
        y_train = train_df[target_col].values  # Convert to numpy array
        
        # Combine for preprocessing
        all_data = pd.concat([
            train_df.drop([target_col], axis=1),
            test_df
        ], axis=0).reset_index(drop=True)  # Reset index after concat
        
        # Handle categorical columns first
        print("\nEncoding categorical columns...")
        encoded_data, _ = encode_categorical_columns(all_data)
        
        # Get feature columns (excluding id if present)
        feature_cols = [col for col in encoded_data.columns if col != 'id']
        
        # Handle numerical missing values
        print("\nImputing numerical missing values...")
        numerical_columns = encoded_data[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        num_imputer = SimpleImputer(strategy='median')
        
        encoded_data[numerical_columns] = num_imputer.fit_transform(encoded_data[numerical_columns])
        
        # Scale features
        print("\nScaling features...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(encoded_data[feature_cols])
        
        # Verify no NaN values remain
        if np.any(np.isnan(scaled_features)):
            raise ValueError("NaN values still present after preprocessing!")
        
        # Split back to train and test
        train_size = len(train_df)
        X_train = scaled_features[:train_size]
        X_test = scaled_features[train_size:]
        
        return X_train, X_test, y_train, scaler, feature_cols
        
    except Exception as e:
        print("\nError occurred during preprocessing:")
        print(f"Exception: {str(e)}")
        if 'train_df' in locals():
            print("\nMissing values in training data:")
            print(train_df.isnull().sum())
        raise

def create_meta_features(X, y, model):
    """Create meta-features from base model predictions"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        model.fit(X[train_idx], y[train_idx])
        meta_features[val_idx] = model.predict(X[val_idx])
    
    return meta_features.reshape(-1, 1)

def select_features(X_train, X_test, y_train, threshold='median'):
    """Select most important features using multiple models"""
    selectors = [
        ('xgb', xgb.XGBRegressor(n_estimators=100)),
        ('lgb', lgb.LGBMRegressor(n_estimators=100)),
        ('gbm', GradientBoostingRegressor(n_estimators=100))
    ]
    
    selected_features = []
    for name, model in selectors:
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(X_train, y_train)
        selected_features.append(selector.get_support())
    
    # Take union of selected features
    final_mask = np.any(selected_features, axis=0)
    return X_train[:, final_mask], X_test[:, final_mask], final_mask

def analyze_feature_importance(model, feature_names):
    """Analyze and print feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices:
            print(f"{feature_names[i]}: {importances[i]:.4f}")
    else:
        print("Model does not have feature_importances_ attribute")

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error"""
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def train_insurance_model(X_train, X_test, y_train, config, feature_cols):
    """Enhanced training with feature selection and meta-features"""
    print("Selecting important features...")
    X_train_selected, X_test_selected, feature_mask = select_features(X_train, X_test, y_train)
    
    # Create polynomial features for selected features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_selected)
    X_test_poly = poly.transform(X_test_selected)
    
    # Optimize XGBoost parameters
    best_xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100
    }
    
    # Create meta-features
    base_lgb = lgb.LGBMRegressor(n_estimators=100)
    meta_features_train = create_meta_features(X_train_selected, y_train, base_lgb)
    base_lgb.fit(X_train_selected, y_train)
    meta_features_test = base_lgb.predict(X_test_selected).reshape(-1, 1)
    
    # Combine original and meta-features
    X_train_enhanced = np.hstack([X_train_selected, meta_features_train])
    X_test_enhanced = np.hstack([X_test_selected, meta_features_test])
    
    # Create enhanced model stack with meta-features
    estimators = [
        ('gbm', GradientBoostingRegressor(**config['params'])),
        ('xgb', xgb.XGBRegressor(**best_xgb_params)),
        ('lgb', lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.01,
            feature_fraction=0.9
        )),
        ('poly_gbm', GradientBoostingRegressor(
            **{**config['params'], 'n_estimators': 50}
        ))
    ]
    
    # Use bagging with enhanced features
    n_bags = 5
    predictions = np.zeros(len(X_test))
    cv_scores = []
    
    for bag in range(n_bags):
        print(f"\nTraining bag {bag + 1}/{n_bags}")
        # Create stacking model
        stack_model = StackingRegressor(
            estimators=estimators,
            final_estimator=LassoCV(),
            cv=5
        )
        
        # Random sample with replacement (bagging)
        indices = np.random.choice(len(X_train_enhanced), len(X_train_enhanced), replace=True)
        X_bag = X_train_enhanced[indices]
        y_bag = y_train[indices]
        
        # Train and predict
        stack_model.fit(X_bag, y_bag)
        bag_pred = stack_model.predict(X_test_enhanced)
        predictions += bag_pred / n_bags
        
        # Calculate CV score for this bag
        cv_score = cross_val_score(
            stack_model, X_bag, y_bag,
            scoring=lambda est, X, y: -rmsle(y, est.predict(X)),
            cv=5
        )
        cv_scores.append(-cv_score.mean())
        
        # Feature importance for this bag
        for name, model in stack_model.named_estimators_.items():
            print(f"\nFeature Importance for {name} (Bag {bag + 1}):")
            analyze_feature_importance(model, feature_cols)
    
    final_cv_rmsle = np.mean(cv_scores)
    print(f"\nFinal Cross-validation RMSLE: {final_cv_rmsle:.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    return stack_model, predictions, final_cv_rmsle, feature_mask
