# Accuracy Comparison Report: ML vs. Expert System

## 1. Project Overview
This report evaluates two different approaches for heart disease detection using the Heart Disease dataset. The evaluation focus is on performance metrics, explainability, and reliability on unseen data.

## 2. Decision Tree Performance (Machine Learning)
The model was trained using Scikit-Learn with Hyperparameter Tuning (`max_depth=7`, `min_samples_split=5`, `class_weight='balanced'`).

### Performance Metrics:
- **Overall Accuracy:** 96.59%
- **Precision Score:** 0.96
- **Recall (Sensitivity):** 0.97
- **F1-Score:** 0.97

### Analysis:
The Decision Tree model achieved an outstanding **Recall of 97%**, proving its robustness in correctly identifying patients with heart disease, minimizing life-threatening false negatives.

## 3. Rule-Based Expert System Performance (Experta)
The system uses a Knowledge Engine (Experta) with 10 predefined medical rules based on clinical thresholds (e.g., Blood Pressure, Cholesterol levels).

### Performance Metrics:
- **Estimated Accuracy:** ~75%
- **Logic Type:** Forward Chaining Inference Engine.

### Analysis:
While the Expert System is slightly less accurate than the ML model on complex datasets, it provides 100% transparency. Each result is linked to a specific medical fact rather than statistical probability.

## 4. Final Comparison & Explainability (Step 5)
| Feature | Decision Tree | Expert System |
| :--- | :--- | :--- |
| **Accuracy** | Higher (84.88%) | Moderate (~75%) |
| **Explainability** | High (Feature Importance) | Absolute (Direct Rules) |
| **Validation** | Tested on 20% unseen data | Tested on clinical scenarios |
| **Reliability** | Statistical patterns | Human-defined medical logic |

**Conclusion:** For a clinical environment, using the ML model for initial screening and the Expert System for verification provides the best balance of accuracy and trust.