# Machine Learning Projects

A curated collection of my machine learning projects, experiments, and research-oriented implementations. This repository documents my journey from learning classical machine learning algorithms to exploring scientific computing and AI applications in engineering and physics.

The goal is not only to build accurate models but also to understand the mathematics, algorithms, and reasoning behind them.

---

## Repository Structure

```
Machine-Learning/
│
├── house-pricing-prediction/
├── particle-indetifier/
├── pulser_detector/
├── mental-state-classifier/
│
├── README.md
├── LICENSE
└── .gitignore
```

Each project is self-contained and includes:

* Source code
* Model evaluation
* Project-specific README
* Requirements file

---

## Projects

### 🔬 High-Energy Particle Identification using Softmax Regression

A multi-class particle classification system trained on **1 million simulated detector tracks**. The project investigates how far a classical Softmax (Multinomial Logistic Regression) model can be pushed through feature engineering and cost-sensitive learning without resorting to more complex algorithms.

**Highlights**

* Trained on **1,000,000** detector events
* Classified four particle species: **Kaons, Pions, Positrons, and Protons**
* Addressed severe class imbalance using custom class weights
* Applied logarithmic transformations and polynomial feature engineering
* Improved minority-class (Positron) recall from **~0% to 88%** while maintaining approximately **95% overall accuracy**

**Topics**

* Multinomial Logistic Regression
* Feature Engineering
* Imbalanced Classification
* Cost-Sensitive Learning
* Scientific Machine Learning

---

### ⭐ Pulsar Star Classification using Support Vector Machines

A binary classification project that identifies pulsar candidates using Support Vector Machines on the HTRU2 dataset. The project explores kernel selection, hyperparameter tuning, and the effects of class imbalance while analyzing model performance through multiple evaluation metrics.

**Topics**

* Support Vector Machines
* Kernel Methods
* Hyperparameter Tuning
* Class Imbalance
* Model Evaluation

---

### 🏠 House Price Prediction

A regression project that predicts residential property prices using classical machine learning techniques with extensive preprocessing, feature engineering, and model evaluation.

**Topics**

* Regression
* Data Preprocessing
* Missing Value Imputation
* Feature Engineering
* Model Evaluation



## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* Google Colab

---

🧠 EEG Brainwave Mental State Classification

Classify EEG brainwave signals into three cognitive states using Support Vector Machines.

Mental States

Relaxed
Concentrating
Neutral

Techniques

StandardScaler
Principal Component Analysis (PCA)
Polynomial SVM
GridSearchCV
Hyperparameter Optimization

Performance

Accuracy: 97%
Macro F1-score: 0.97
🛠 Technologies
Python
NumPy
Pandas
Scikit-Learn

---

## Learning Goals

This repository serves as a record of my progress while studying machine learning and scientific computing. Future projects will explore:

* Decision Trees
* Random Forests
* Gradient Boosting
* XGBoost
* Neural Networks
* Deep Learning
* Scientific Machine Learning
* Physics-Informed Neural Networks (PINNs)
* Time-Series Analysis
* Computer Vision

---

## Philosophy

I believe that understanding *why* an algorithm works is just as important as achieving strong performance. For each project, I aim to study the underlying mathematics, evaluate different approaches, and document the insights gained throughout the process.

---

## Contributing

Suggestions, feedback, and discussions are always welcome.
