# Higgs Boson Classification with Gradient Boosting

A machine learning project exploring the classification of simulated particle-physics events, with the goal of distinguishing **signal events associated with Higgs boson production** from background events.

The project started with a relatively simple classification model and achieved approximately **65% accuracy**. From there, I experimented with feature selection, dimensionality reduction, kernels, ensemble methods, and hyperparameter optimization to understand what actually improved performance.

The best final model achieved:

> **72.28% accuracy**

on a held-out test set of **9,999 events**.

---

## Results

### Final Classification Report

|        Class | Precision | Recall | F1-score |   Support |
| -----------: | --------: | -----: | -------: | --------: |
|            0 |      0.71 |   0.69 |     0.70 |     4,700 |
|            1 |      0.73 |   0.75 |     0.74 |     5,299 |
| **Accuracy** |           |        | **0.72** | **9,999** |
|    Macro Avg |      0.72 |   0.72 |     0.72 |     9,999 |
| Weighted Avg |      0.72 |   0.72 |     0.72 |     9,999 |

The final Gradient Boosting classifier reached **72.28% accuracy**, improving substantially over the initial ~65% baseline.

---

## Models Explored

Rather than immediately choosing a single algorithm, I compared several approaches to understand how different modeling assumptions affected the classification problem.

Experiments included:

* Baseline classification model
* Bagging
* Random Forest
* Gradient Boosting
* XGBoost
* PCA
* Kernel-based methods
* Feature selection
* Hyperparameter optimization using `GridSearchCV`

The best hyperparameters for the ensemble models were determined using cross-validated grid search rather than manually selecting parameters based on the test set.

### Model Performance

| Model / Approach  |   Accuracy |
| ----------------- | ---------: |
| Initial baseline  |       ~65% |
| XGBoost           |    ~70.22% |
| Gradient Boosting | **72.28%** |

The exact performance depends on the preprocessing and feature configuration used in each experiment.

---

## A Surprising Lesson About Feature Selection

One of the most interesting parts of this project was experimenting with feature correlation.

Initially, removing features with low correlation seemed like a reasonable dimensionality-reduction strategy.

However, the experiments showed that this assumption was not reliable.

Some features with relatively weak individual correlation with the target still contributed useful information to the classifier. Removing them could noticeably affect performance.

This was particularly interesting in the context of particle-physics data.

### Why?

Correlation measures the strength of a **simple pairwise relationship** between an individual feature and the target.

A machine-learning model, however, can exploit:

* nonlinear relationships
* interactions between features
* combinations of weak signals
* information that only becomes useful when considered alongside other variables

So a feature can appear relatively unimportant when examined independently while still contributing predictive information to an ensemble model.

This became one of the main lessons of the project:

> **Low individual correlation does not necessarily mean low predictive value.**

Feature selection therefore needs to be evaluated experimentally rather than relying solely on correlation thresholds.

---

## Why Gradient Boosting?

Gradient Boosting ultimately performed better than the other approaches I tested.

The model builds an ensemble of weak learners sequentially, with later learners focusing on errors made by earlier ones.

This makes it particularly useful for structured tabular data where the relationship between features and the target may be nonlinear and involve feature interactions.

After hyperparameter optimization with `GridSearchCV`, Gradient Boosting achieved the best observed test performance of **72.28%**.

---

## Experimental Process

The project followed an iterative workflow:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Baseline Model
     ↓
~65% Accuracy
     ↓
Feature Analysis
     ↓
Feature Selection Experiments
     ↓
PCA / Kernel Experiments
     ↓
Ensemble Models
     ↓
Hyperparameter Optimization
     ↓
Model Comparison
     ↓
Gradient Boosting
     ↓
72.28% Accuracy
```

The objective wasn't simply to maximize the final accuracy.

A major goal was to understand **why** particular preprocessing and modeling choices helped or hurt performance.

---

## Key Takeaways

### 1. Baselines matter

The initial model achieved approximately 65% accuracy.

Without establishing that baseline, it would be difficult to determine whether later experiments were actually improving the system.

### 2. More complex models don't automatically win

XGBoost plateaued around **70.22%**, while Gradient Boosting reached **72.28%** in these experiments.

Using a more popular or sophisticated algorithm does not guarantee better performance on a particular dataset.

### 3. Feature selection is not just "remove weak features"

The experiments showed that low correlation with the target does not necessarily mean that a feature is useless.

Feature interactions can contain information that isn't visible through simple correlation analysis.

### 4. Hyperparameter optimization helps, but doesn't replace experimentation

`GridSearchCV` helped identify strong hyperparameter configurations, but the larger improvements came from understanding the dataset and systematically comparing different approaches.

### 5. Accuracy isn't the whole story

The final model achieved relatively balanced performance across the two classes:

* Class 0 F1: **0.70**
* Class 1 F1: **0.74**

This suggests that the model isn't simply performing well by heavily favoring one class.

---

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* XGBoost
* Jupyter Notebook

---

## Future Improvements

There is still considerable room for experimentation.

Potential next steps include:

* More systematic feature engineering
* Feature interaction analysis
* Bayesian hyperparameter optimization
* More extensive XGBoost tuning
* LightGBM / CatBoost comparison
* Calibration and probability analysis
* ROC-AUC and Precision-Recall analysis
* SHAP-based model interpretability
* Error analysis of misclassified events
* Investigating class-dependent decision thresholds
* More robust cross-validation

The goal of the next iteration would be to understand **where the remaining classification errors come from**, rather than simply searching for a higher accuracy number.

---

## Conclusion

This project started with a simple classification approach achieving roughly **65% accuracy** and progressed through multiple experiments to a final **72.28% accuracy** Gradient Boosting model.

More importantly, the project reinforced an important lesson about applied machine learning:

> **The best model isn't always the most complicated model, and the best feature isn't always the one with the highest correlation.**

In scientific datasets, understanding the structure of the data can be just as important as choosing the algorithm.
