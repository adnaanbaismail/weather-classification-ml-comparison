# 🌦️ Weather Classification: Multi-Model Machine Learning Benchmark

A comparative machine learning study predicting weather conditions (**Rainy**, **Cloudy**, **Sunny**, **Snowy**) from meteorological and geographical features. This project evaluates **9 classification algorithms**—spanning individual baseline classifiers and ensemble architectures—achieving a peak accuracy of **91.25%** with a Bagging Classifier.

---

## 📊 Model Performance Summary

All models were evaluated on a 20% holdout test split (`random_state=42`) after ordinal encoding and feature standardization:

| Rank | Model | Type | Test Accuracy |
| :--- | :--- | :--- | :---: |
| 🥇 **1** | **Bagging Classifier (60 Trees)** | Ensemble (Bagging) | **91.25%** |
| 🥈 **2** | **Gradient Boosting** | Ensemble (Boosting) | **91.10%** |
| 🥉 **3** | **Decision Tree** | Single Tree | **91.02%** |
| **4** | **Stacking Classifier** (`DT + SVC + NB`) | Ensemble (Stacking) | **90.87%** |
| **5** | **Soft Voting** (`LR + DT + NB`) | Ensemble (Voting) | **89.20%** |
| **6** | **k-Nearest Neighbors (`k=5`)** | Instance-Based | **89.17%** |
| **7** | **Hard Voting** (`LR + DT + NB`) | Ensemble (Voting) | **87.42%** |
| **8** | **Gaussian Naive Bayes** | Probabilistic | **86.14%** |
| **9** | **Logistic Regression** | Linear | **85.45%** |

---

## 🔍 Key Insights

* **Tree-Based Models Dominate:** Weather classification relies heavily on non-linear thresholds (e.g., sub-zero temperatures for *Snowy*, high humidity and cloud cover for *Rainy*). Tree architectures (`Bagging`, `Gradient Boosting`, `Decision Tree`) naturally capture these decision boundaries, all surpassing **91%** accuracy.
* **Variance Reduction via Bagging:** Aggregating 60 Decision Trees reduces single-tree overfitting and achieves the highest overall accuracy (**91.25%**).
* **Ensemble Diversity Trade-offs:** Combining weaker linear/probabilistic classifiers (`Logistic Regression` at 85.45%, `Naive Bayes` at 86.14%) with a strong `Decision Tree` (91.02%) pulls down `Hard Voting` (87.42%) when the two weaker models outvote the stronger one.

---

## 🛠️ Pipeline & Methodology

1. **Exploratory Data Analysis (EDA):**
   * Inspected feature data types, unique categorical levels, and numerical distributions.
   * Visualized numerical feature spreads and outliers using boxplots (`Matplotlib` & `Seaborn`).
2. **Preprocessing:**
   * **Categorical Encoding:** Applied `OrdinalEncoder` to `Cloud Cover`, `Season`, `Location`, and `Weather Type`.
   * **Train-Test Split:** 80% training / 20% testing (`random_state=42`).
   * **Feature Scaling:** Standardized numerical features using `StandardScaler`.
3. **Model Training & Evaluation:**
   * Trained 4 baseline classifiers (`LR`, `GaussianNB`, `DecisionTree`, `KNN`) and 5 ensemble methods (`Bagging`, `GradientBoosting`, `Hard Voting`, `Soft Voting`, `Stacking`).
   * Evaluated using `accuracy_score`, full `classification_report` (precision, recall, F1-score), and a sorted comparison bar chart.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
