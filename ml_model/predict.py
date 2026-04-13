import joblib
import pandas as pd
import os

def run_inference(patient_data):
    # 1. Setup paths using a slightly different style
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    path_model = os.path.join(project_dir, 'ml_model', 'heart_model.pkl')
    path_cols = os.path.join(project_dir, 'ml_model', 'model_columns.pkl')
    path_scaler = os.path.join(project_dir, 'ml_model', 'scaler.pkl')

    # 2. Load the saved artifacts
    dt_model = joblib.load(path_model)
    expected_features = joblib.load(path_cols)
    min_max_scaler = joblib.load(path_scaler)
    
    # 3. Initialize an empty dataframe matching the model's exact expected columns
    final_input_df = pd.DataFrame(0.0, index=[0], columns=expected_features)
    
    # 4. Process and scale numerical columns
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    raw_numerics = pd.DataFrame([patient_data])[numeric_features]
    scaled_numerics = min_max_scaler.transform(raw_numerics)[0]
    
    for index, col_name in enumerate(numeric_features):
        if col_name in expected_features:
            final_input_df.at[0, col_name] = scaled_numerics[index]

    # 5. Process binary features seamlessly
    binary_features = ['sex', 'fbs', 'exang']
    for b_col in binary_features:
        if b_col in expected_features and b_col in patient_data:
            final_input_df.at[0, b_col] = float(patient_data[b_col])

    # 6. Handle One-Hot Encoded categorical variables dynamically
    category_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    for cat_col in category_features:
        if cat_col in patient_data:
            category_val = patient_data[cat_col]
            
            # Formulate possible column names (handling both int and float formats)
            str_int_format = f"{cat_col}_{int(category_val)}"
            str_float_format = f"{cat_col}_{float(category_val)}"
            
            if str_int_format in expected_features:
                final_input_df.at[0, str_int_format] = 1.0
            elif str_float_format in expected_features:
                final_input_df.at[0, str_float_format] = 1.0
                
    # 7. Generate predictions
    predicted_class = dt_model.predict(final_input_df)[0]
    prediction_probs = dt_model.predict_proba(final_input_df)[0]
    
    return int(predicted_class), prediction_probs

# Test block
if __name__ == "__main__":
    # Test with a sample data point 
    test_patient = {
        'age': 62, 'sex': 1, 'cp': 0, 'trestbps': 160, 'chol': 280, 
        'fbs': 1, 'restecg': 1, 'thalach': 110, 'exang': 1, 
        'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 3
    }
    
    print("--> Processing patient data and running the Decision Tree Model...")
    outcome, probabilities = run_inference(test_patient)
    
    result_label = "High Risk" if outcome == 0 else "Low Risk"
    
    print("\n" + "*"*45)
    print(f"  DIAGNOSIS RESULT: {result_label.upper()}  ")
    print("*"*45)
    print(f"-> Probability of being Healthy : {probabilities[1]*100:.1f}%")
    print(f"-> Probability of Heart Disease : {probabilities[0]*100:.1f}%\n")