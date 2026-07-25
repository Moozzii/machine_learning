# 🔬 High-Energy Particle Identification Classifier

[![Open In Colab](https://google.com)](https://google.com)

A multi-class Softmax classifier built with Scikit-Learn and trained on **1,000,000 rows** of simulated high-energy physics detector tracks. The pipeline reliably classifies four overlapping subatomic particles: **Pions, Protons, Kaons, and Positrons**.

## 🚀 The Engineering Challenge
In true experimental environments, background particles (**Pions**) overwhelmingly flood the detector sensors, making target signal carriers (**Positrons**) extremely rare. 

* **The Class Imbalance**: Positrons account for only **0.3%** of the dataset (3,026 out of 1,000,000 instances).
* **The Baseline Failure**: A standard unweighted Softmax model suffered from severe gradient swamping. It yielded a misleadingly high global accuracy (~95.7%) but missed almost all minority signals, catching only **1 out of 3,026 true positrons** (0.03% Recall).
* **The Physics Overlap**: Raw kinematic properties like momentum (`p`) and velocity (`beta`) closely overlap between pions and positrons. A standard flat linear classifier cannot split these boundaries natively.

## 🛠️ Machine Learning Pipeline Architecture
To solve this without stepping away from Logistic Regression / Softmax, the following data engineering pipeline was established:

1. **Skew-Correction Transformers**: Heavy right-sided distributions in physical energy deposition features (`ein`, `eout`) were pulled into symmetric bell curves using a `FunctionTransformer(np.log1p)`.
2. **Feature Boundary Expansion**: Polynomial interactions (`degree=4`) were injected to mathematically bend the feature space, allowing a linear algorithm to trace non-linear, curving particle tracks.
3. **Loss Function Modification**: Custom class weights inversely proportional to sample density were written into the cross-entropy loss function. This penalized the optimizer **50x harder** for missing a positron than a pion.


## 📊 Performance Metrics

### Optimized Confusion Matrix
```text
[[ 40206   4977    132   1058]  <- Kaon
 [ 23993 524216   9078   4569]  <- Pion
 [     3    368   2655      0]  <- Positron
 [  8069    248    184 380244]] <- Proton
```

### Production Classification Report
```text
              precision    recall  f1-score   support

        kaon       0.56      0.87      0.68     46373
        pion       0.99      0.93      0.96    561856
    positron       0.22      0.88      0.35      3026
      proton       0.99      0.98      0.98    388745

    accuracy                           0.95   1000000
   macro avg       0.68      0.91      0.74   1000000
weighted avg       0.97      0.95      0.95   1000000
```
* **Engineering Trade-off Note**: Positron recall skyrocketed from **0.03% to 88%**. In high-energy particle physics, catching rare events is prioritized over false alarms; downstream triggers can filter out the remaining 9,078 pion leaks.

## 📦 How to Use the Saved Pipeline
The complete, integrated preprocessing and classifier model is serialized in a single workspace bundle.

```python
import joblib

# Load the complete deployment pipeline
model = joblib.load('detector_particle_pipeline.joblib')

# Predict raw array shapes smoothly without manual data processing steps
# Input shape: [p, theta, beta, nphe, ein, eout]
predictions = model.predict(X_test_raw)
```
