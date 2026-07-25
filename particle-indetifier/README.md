# 🔬 High-Energy Particle Identification using Softmax Regression

A multi-class particle classifier built with **Scikit-Learn** and trained on **1,000,000 simulated detector tracks**. The model identifies four types of particles from detector measurements:

- Kaons
- Pions
- Positrons
- Protons

The objective was **not** to build the most accurate classifier possible, but to explore how far a classical Softmax (multinomial logistic regression) model can be pushed through feature engineering and proper handling of highly imbalanced data.

---

## 📌 Project Motivation

Particle detectors generate enormous amounts of data, but not every particle appears with the same frequency. Some particles are extremely common while others are very rare.

In this dataset:

| Particle | Samples |
|----------|---------:|
| Pion | 561,856 |
| Proton | 388,745 |
| Kaon | 46,373 |
| Positron | 3,026 |

Positrons represent only **0.3%** of the entire dataset.

A standard Softmax classifier achieved high overall accuracy simply by predicting the majority classes, but almost completely failed to recognize positrons.

The challenge was to improve minority-class detection **without replacing Logistic Regression with a more powerful model.**

---

# 🚀 Approach

The complete workflow was implemented as a Scikit-Learn Pipeline.

### 1. Log Transformation

Energy deposition features (`ein`, `eout`) were heavily skewed.

A `log1p` transformation was applied to make their distributions more suitable for linear classification.

---

### 2. Polynomial Feature Engineering

Since Softmax Regression is a linear classifier, fourth-degree polynomial features were introduced to allow nonlinear decision boundaries.

This significantly improved separation between overlapping particle classes.

---

### 3. Class Weighting

Because positrons are extremely rare, the optimizer naturally focused on the dominant pion class.

Custom class weights were added so that mistakes on minority classes contributed more heavily to the loss function.

This shifted the optimization objective from maximizing accuracy to improving minority-class recall.

---

## 📊 Results

### Confusion Matrix

```text
[[ 40206   4977    132   1058]
 [ 23993 524216   9078   4569]
 [     3    368   2655      0]
 [  8069    248    184 380244]]
```

### Classification Report

```text
              precision    recall  f1-score

Kaon          0.56       0.87      0.68
Pion          0.99       0.93      0.96
Positron      0.22       0.88      0.35
Proton        0.99       0.98      0.98

Overall Accuracy: 95%
```

---

## 🎯 Key Insight

The original unweighted Softmax model achieved approximately **95.7% accuracy**, but detected almost none of the positrons.

After feature engineering and class balancing:

- Positron Recall improved from **0.03% → 88%**
- Overall accuracy remained around **95%**

Although precision for positrons decreased, this trade-off is often acceptable in high-energy physics where missing a rare event is generally more costly than generating additional false positives.

---

## 💻 Technologies

- Python
- NumPy
- Pandas
- Scikit-Learn
- Joblib

---

## 📦 Model Deployment

The complete preprocessing pipeline and trained classifier are serialized into a single file.

```python
import joblib

model = joblib.load("detector_particle_pipeline.joblib")

predictions = model.predict(X_test_raw)
```

---

## 📚 What I Learned

This project reinforced several important machine learning concepts:

- Accuracy can be misleading for highly imbalanced datasets.
- Precision and recall often provide a more meaningful evaluation than overall accuracy.
- Feature engineering can significantly improve the performance of linear models.
- Class weighting changes the optimization objective without changing the underlying algorithm.
- Understanding the problem domain is essential when selecting evaluation metrics.

---

## Future Improvements

- Compare Softmax Regression with XGBoost and Random Forests.
- Experiment with SMOTE and other resampling techniques.
- Explore calibrated probability estimates.
- Investigate neural network classifiers for comparison.