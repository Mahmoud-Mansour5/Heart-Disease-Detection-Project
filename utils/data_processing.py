import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def process_data():
    # Set up file paths based on the project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path     = os.path.join(base_dir, 'data', 'raw_data.csv')
    cleaned_data_path = os.path.join(base_dir, 'data', 'cleaned_data.csv')
    scaler_path       = os.path.join(base_dir, 'ml_model', 'scaler.pkl')

    # 1. Load Dataset
    try:
        df = pd.read_csv(raw_data_path)
        print("loaded successfully.")
    except FileNotFoundError:
        print(f" Error: {raw_data_path} not found.")
        return

    # 2. Handle Missing Values
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True) 
    print(" Missing values handled .")

    # Define columns
    # Note: Columns like sex, fbs, and exang are binary (0/1), so no scaling is needed.
    numerical_cols  = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

    # 3. Normalize Data using MinMaxScaler
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Ensure the ml_model directory exists before saving the scaler to avoid errors
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f" Numerical features normalized. Scaler saved at: {scaler_path}")

    # 4. Encode Categorical Variables (Scikit-Learn One-Hot Encoding)
    ohe_encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    encoded_array = ohe_encoder.fit_transform(df[categorical_cols])
    
    encoded_col_names = ohe_encoder.get_feature_names_out(categorical_cols)
    
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df.index)
    
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    df[encoded_col_names] = df[encoded_col_names].astype(int)
    
    print("✅ Categorical variables encoded using Scikit-Learn OneHotEncoder.")
    # 5. Feature Selection 
    correlation_matrix = df.corr()
    target_corr = correlation_matrix['target'].abs().sort_values(ascending=False)
    
    print("\n the top 10 most influential features based on correlation with the target variable:")
    print(target_corr.head(10))
    threshold = 0.05
    selected_features = target_corr[target_corr > threshold].index.tolist()
    
    df = df[selected_features]
    print(f"\n Feature selection applied. Kept {len(selected_features)} features out of {len(target_corr)}.")

    ###################### Save Cleaned Data
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    df.to_csv(cleaned_data_path, index=False)
    print(" Cleaned dataset saved.")

if __name__ == "__main__":
    process_data()