# Pulsar Star Classification using Support Vector Machines

A machine learning project that classifies pulsar stars using Support Vector Machines (SVM) on the HTRU2 dataset. This project explores different SVM kernels, hyperparameter tuning, and techniques for handling class imbalance while evaluating performance through multiple classification metrics.

---

## Overview

Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation. Detecting pulsars is an important task in radio astronomy, but modern telescopes generate enormous amounts of candidate signals, making manual inspection impractical.

This project applies Support Vector Machines to distinguish pulsar candidates from non-pulsar signals using the HTRU2 dataset.

---

## Dataset

**Dataset:** HTRU2 (High Time Resolution Universe Survey)

- 17,898 observations
- 8 numerical features
- Binary classification
- Class imbalance (~90% non-pulsars, ~10% pulsars)

The features are statistical summaries extracted from integrated pulse profiles and DM-SNR curves.

---

## Project Workflow

- Data loading and exploration
- Data preprocessing
- Feature scaling using StandardScaler
- Train-test split
- Training Support Vector Machine classifiers
- Hyperparameter tuning
- Evaluation using:
  - Classification Report
  - Confusion Matrix
  - Accuracy
  - Precision
  - Recall
  - F1-score

---

## Models Explored

- Linear SVM
- Polynomial Kernel
- RBF Kernel
- Sigmoid Kernel

Hyperparameters explored include:

- C
- Polynomial Degree
- Class Weights

---

## Results

Best model achieved approximately:

| Metric | Score |
|---------|-------|
| Accuracy | **98.45%** |
| Precision (Pulsar) | **93%** |
| Recall (Pulsar) | **89%** |
| F1 Score | **91%** |

The experiments showed that increasing the regularization parameter beyond approximately **C = 15** resulted in negligible improvements, indicating that the classifier had reached a performance plateau on the available feature space.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

---

## Repository Structure

```
├── pulser_detector_model.pkl
├── Pulsar_Detection.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Future Improvements

- GridSearchCV for automated hyperparameter optimization
- ROC and Precision-Recall curves
- Cross-validation
- PCA visualization
- Comparison with Random Forest and XGBoost
- Analysis of Support Vectors
- Feature importance using permutation methods

---

## License

This project is released under the MIT License.