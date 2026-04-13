import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_robust_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'cleaned_data.csv')
    model_path = os.path.join(base_dir, 'ml_model', 'heart_model.pkl')
    columns_path = os.path.join(base_dir, 'ml_model', 'model_columns.pkl')


    df = pd.read_csv(data_path)

    #  Separate features (X) and target (y)
    X, y = df.drop('target', axis=1), df['target']
    
    #  Split Data: 80% Train, 20% Test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    #  Hyperparameter Tuning using GridSearchCV
    print(" Tuning hyperparameters ")
    param_grid = {
        'max_depth': [3, 4, 5],                
        'min_samples_split': [10, 20, 30],  
        'min_samples_leaf': [10, 15, 20],
        'class_weight': ['balanced']
    }
    
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"  Hyperparameters found: {grid_search.best_params_}")
    
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\n Model Performance on Test Set:")
    
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
   

if __name__ == "__main__":
    train_robust_model()