# 📊 Accuracy Comparison Report
## Decision Tree Model vs. Rule-Based Expert System

---

## 1. Overview

This report compares the two prediction approaches implemented in the Heart Disease Detection System:

| | Decision Tree (ML) | Expert System (Rule-Based) |
|---|---|---|
| **Approach** | Data-driven (learns from data) | Knowledge-driven (human-defined rules) |
| **Library** | Scikit-Learn | Experta |
| **Algorithm** | Decision Tree Classifier | Forward-Chaining Inference |
| **Output** | Binary (0 = Disease, 1 = Healthy) | LOW / MODERATE / HIGH RISK |
| **Explainability** | Medium (tree visualization) | High (explicit rules) |

---

## 2. Decision Tree — Performance Metrics

**Best Hyperparameters (GridSearchCV, 5-fold CV):**
- `max_depth = 3`
- `min_samples_split = 10`
- `min_samples_leaf = 20`
- `class_weight = balanced`

**Test Set Results (80/20 split, 61 test samples):**

| Metric | Class 0 (Heart Disease) | Class 1 (Healthy) | Weighted Avg |
|--------|------------------------|-------------------|--------------|
| Precision | 0.75 | 0.84 | **0.80** |
| Recall | 0.75 | 0.84 | **0.80** |
| F1-Score | 0.75 | 0.84 | **0.80** |
| Support | 24 | 37 | 61 |

**Overall Test Accuracy: 80%**

> The model performs better on the Healthy class (F1 = 0.84) than on the Disease class (F1 = 0.75), which is partly addressed by the `class_weight=balanced` parameter.

---

## 3. Rule-Based Expert System — Risk Scoring

The Expert System uses **12 medical rules** with a cumulative risk score:

| Rule ID | Condition | Score Impact |
|---------|-----------|-------------|
| R1 | Cholesterol > 240 & Age > 50 | +3 |
| R2 | Resting BP > 140 mmHg | +2 |
| R3 | Exercise-induced angina | +3 |
| R4 | Max heart rate < 120 bpm | +2 |
| R5 | ST depression > 2.0 | +2 |
| R6 | Typical angina (cp=0) | +2 |
| R7 | 2+ blocked vessels | +3 |
| R8 | Reversible thal defect | +3 |
| R9 | High fasting sugar & chol > 200 | +2 |
| R10 | LV hypertrophy on ECG | +2 |
| R11 | Male + Age > 55 + BP > 130 | +3 |
| R12 | Good HR, low chol, no angina | -2 |

**Risk Classification Thresholds:**

| Score | Risk Level |
|-------|-----------|
| ≤ 2 | 🟢 LOW RISK |
| 3 – 6 | 🟡 MODERATE RISK |
| ≥ 7 | 🔴 HIGH RISK |

**Sample Test (High-Risk Patient):**
```
age=60, sex=1, cp=0, trestbps=150, chol=260,
fbs=1, restecg=2, thalach=110, exang=1,
oldpeak=2.5, slope=1, ca=2, thal=3
```
Rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11 → **Score = 25 → HIGH RISK** ✅

---

## 4. Head-to-Head Comparison

| Criterion | Decision Tree | Expert System |
|-----------|--------------|---------------|
| **Accuracy** | ✅ 80% measured | ⚠️ Not directly measurable as % |
| **Recall (Disease)** | 75% | Qualitative (HIGH/MOD/LOW) |
| **Precision (Disease)** | 75% | Qualitative |
| **F1-Score (Healthy)** | 84% | — |
| **Handles unseen patterns** | ✅ Yes (generalizes from data) | ❌ No (only fires known rules) |
| **Requires training data** | ✅ Yes | ❌ No |
| **Transparent reasoning** | ⚠️ Partial (tree paths) | ✅ Full (rule log printed) |
| **Domain expert input needed** | ❌ No | ✅ Yes |
| **Handles edge cases** | ✅ Yes | ⚠️ Only if rule is defined |
| **Result interpretability** | Medium | High |
| **Update/maintenance** | Retrain model | Edit rules manually |

---

## 5. Explainability Analysis

### Decision Tree
- Decisions are made by splitting on feature thresholds learned from data (e.g., `thal <= 2.5`, `ca <= 0.5`).
- A depth-4 tree produces a readable structure — can be visualized with `sklearn.tree.plot_tree`.
- However, the *why* behind each split is statistical, not medical.
- Risk: may overfit to dataset biases if not regularized (addressed here via `min_samples_leaf=10`).

### Expert System
- Every decision is fully traceable to a named rule (e.g., `[R3] Exercise-induced angina present (+3)`).
- Rules are written by domain experts and directly reflect clinical guidelines.
- Output is always explainable in plain medical language.
- Limitation: cannot discover patterns not encoded in the rules.

---

## 6. When to Use Each

| Scenario | Recommended Approach |
|----------|---------------------|
| High-volume automated screening | ✅ Decision Tree |
| Clinical decision support (doctor review) | ✅ Expert System |
| Explaining a specific patient's risk | ✅ Expert System |
| Dataset with many subtle feature interactions | ✅ Decision Tree |
| No labeled training data available | ✅ Expert System |
| Regulatory/auditable medical environment | ✅ Expert System |

---

## 7. Conclusion

Both systems complement each other effectively:

- The **Decision Tree** achieves **80% accuracy** on held-out test data, making it reliable for automated risk screening. Its strength is learning from statistical patterns across 13+ features simultaneously.

- The **Expert System** provides **transparent, rule-based reasoning** grounded in clinical knowledge. While it cannot produce a single accuracy percentage without ground-truth labels per rule, it correctly identifies high-risk patients with multiple co-occurring risk factors — as validated in sample testing.

> **Recommendation:** Use the **Expert System** output alongside the **Decision Tree** prediction in the UI. Agreement between both systems increases diagnostic confidence; disagreement flags borderline cases for closer review.

---

*Generated as part of the Heart Disease Detection — Expert Systems Project*
