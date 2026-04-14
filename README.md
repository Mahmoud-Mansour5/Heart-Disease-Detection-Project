# 🫀 Heart Disease Detection System

A hybrid AI system for heart disease risk prediction combining a **Rule-Based Expert System** (Experta) and a **Machine Learning Model** (Decision Tree Classifier).

---

## 📁 Project Structure

```
Heart_Disease_Detection/
│── data/
│   ├── raw_data.csv               # Original dataset
│   └── cleaned_data.csv           # Preprocessed dataset
│── notebooks/
│   ├── data_analysis.ipynb        # EDA & visualizations
│   └── model_training.ipynb       # Model training notebook
│── rule_based_system/
│   ├── rules.py                   # Experta rules & knowledge base
│   └── expert_system.py           # Expert system engine & UI bridge
│── ml_model/
│   ├── train_model.py             # Decision Tree training script
│   └── predict.py                 # Inference script
│── utils/
│   └── data_processing.py         # Data cleaning & preprocessing
│── reports/
│   └── accuracy_comparison.md     # Model vs Expert System comparison
│── ui/
│   └── app.py                     # Streamlit web interface
│── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Heart_Disease_Detection.git
cd Heart_Disease_Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the Dataset
Place your raw heart disease CSV file at:
```
data/raw_data.csv
```
The dataset should contain these columns:
`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`

---

## 🚀 Usage

### Step 1 — Preprocess the Data
```bash
python utils/data_processing.py
```
- Handles missing values and duplicates
- Applies MinMaxScaler on numerical features
- One-Hot Encodes categorical variables
- Performs feature selection via correlation analysis
- Saves `data/cleaned_data.csv` and `ml_model/scaler.pkl`

### Step 2 — Train the ML Model
```bash
python ml_model/train_model.py
```
- Runs GridSearchCV for hyperparameter tuning
- Trains a Decision Tree Classifier (80/20 split)
- Saves `ml_model/heart_model.pkl` and `ml_model/model_columns.pkl`

### Step 3 — Run Inference (ML Model)
```bash
python ml_model/predict.py
```

### Step 4 — Run the Expert System
```bash
python rule_based_system/expert_system.py
```

### Step 5 — Launch the Streamlit UI
```bash
streamlit run ui/app.py
```

---

## 🧠 System Components

### Rule-Based Expert System (Experta)
Uses a knowledge base of **12 medical rules** to assess risk:

| Rule | Condition | Risk Points |
|------|-----------|-------------|
| R1 | Cholesterol > 240 & Age > 50 | +3 |
| R2 | Resting BP > 140 mmHg | +2 |
| R3 | Exercise-induced angina present | +3 |
| R4 | Max heart rate < 120 bpm | +2 |
| R5 | ST depression (oldpeak) > 2.0 | +2 |
| R6 | Typical angina (cp = 0) | +2 |
| R7 | 2+ major vessels blocked | +3 |
| R8 | Reversible thalassemia defect | +3 |
| R9 | High fasting sugar & cholesterol > 200 | +2 |
| R10 | LV hypertrophy on ECG | +2 |
| R11 | Male, Age > 55 & BP > 130 | +3 |
| R12 | Good HR, low cholesterol, no angina | -2 |

**Risk Levels:**
- Score ≤ 2 → 🟢 LOW RISK
- Score 3–6 → 🟡 MODERATE RISK
- Score ≥ 7 → 🔴 HIGH RISK

### Decision Tree Classifier (Scikit-Learn)
- **Best Hyperparameters:** `max_depth=4`, `min_samples_split=30`, `min_samples_leaf=10`, `class_weight=balanced`
- **Test Accuracy:** 80%
- **F1-Score (Disease class):** 0.84

---

## 📊 Model Performance

| Metric | Class 0 (Disease) | Class 1 (Healthy) | Overall |
|--------|-------------------|-------------------|---------|
| Precision | 0.75 | 0.84 | 0.80 |
| Recall | 0.75 | 0.84 | 0.80 |
| F1-Score | 0.75 | 0.84 | 0.80 |
| Accuracy | — | — | **80%** |

---

## 📦 Requirements

```
pandas
scikit-learn
joblib
experta
streamlit
seaborn
matplotlib
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- The `experta` library requires a Python compatibility fix for `collections.Mapping` (already handled in `rules.py`).
- All trained artifacts (model, scaler, columns) are saved in `ml_model/` after running the training pipeline.
- The system uses the **UCI Heart Disease Dataset** (Cleveland).

---

## 👥 Contributors

Built as part of an Expert Systems university project.
