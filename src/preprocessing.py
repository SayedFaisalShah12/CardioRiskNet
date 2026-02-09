import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import os

def load_and_preprocess_data():
    print("Fetching UCI Heart Disease dataset...")
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 

    # Combine for cleaning
    df = pd.concat([X, y], axis=1)
    
    # Check for missing values
    print(f"Initial shape: {df.shape}")
    print("Missing values per column:\n", df.isnull().sum())
    
    # Simple imputation: Drop rows with missing values for this demo
    df = df.dropna()
    print(f"Shape after dropping missing values: {df.shape}")

    # Feature Engineering/Selection
    # In a real scenario, we might perform more complex encoding for categorical variables.
    # The UCI dataset mostly has numeric or pre-encoded categories.
    
    X = df.drop('num', axis=1) # 'num' is the target in this version (0=no, 1-4=yes)
    y = (df['num'] > 0).astype(int) # Binary classification: 0 (No) vs 1 (Yes)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    os.makedirs('data', exist_ok=True)
    np.save('data/X_train.npy', X_train_scaled)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)

    print("Data preprocessing complete. Saved to 'data/' folder.")

if __name__ == "__main__":
    load_and_preprocess_data()
