import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os

def load_and_preprocess_data():
    print("Loading UCI Heart Disease dataset from local CSV...")
    
    # Column names based on UCI documentation
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    
    # Load data, handling '?' as missing values
    df = pd.read_csv('data/heart_disease.csv', names=columns, na_values='?')
    
    # Check for missing values
    print(f"Initial shape: {df.shape}")
    print("Missing values per column:\n", df.isnull().sum())
    
    # Simple imputation: Drop rows with missing values
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    X = df.drop('num', axis=1) # 'num' is the target (0=no, 1-4=yes)
    y = (df['num'] > 0).astype(int) # Binary classification

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    os.makedirs('data', exist_ok=True)
    import joblib
    joblib.dump(scaler, 'models/scaler.joblib')
    
    np.save('data/X_train.npy', X_train_scaled)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)

    print("Data preprocessing complete. Scaler saved to 'models/scaler.joblib'.")

if __name__ == "__main__":
    load_and_preprocess_data()
