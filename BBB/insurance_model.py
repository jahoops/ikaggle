import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess():
    # Load insurance dataset
    df = pd.read_csv('insurance.csv')
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'])
    
    return df

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, rmse, r2, X_test, y_test, y_pred

def main():
    df = load_and_preprocess()
    
    # Separate features and target
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    model, scaler, rmse, r2, X_test, y_test, y_pred = train_model(X, y)
    
    print(f"RMSE: ${rmse:.2f}")
    print(f"R2 Score: {r2:.3f}")

if __name__ == "__main__":
    main()
