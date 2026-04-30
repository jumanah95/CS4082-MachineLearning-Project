# Hospital Readmission Prediction for Diabetic Patients

## CS4082 – Machine Learning Final Project

### Team Members

* Aseel Bajaber
* Jumanah Al-Nahdi

### Supervisor

Dr. Naila Marir

---

## Project Overview

Hospital readmission within 30 days is a major healthcare challenge, especially for diabetic patients. Frequent readmissions increase hospital costs, worsen patient outcomes, and indicate gaps in follow-up care.

This project aims to predict early hospital readmission risk using Machine Learning on the Diabetes 130-US Hospitals dataset. The project focuses on handling noisy real-world medical records under adversarial conditions using label noise.

The goal is to help hospitals identify high-risk patients before discharge and support better clinical decision-making.

---

## Dataset Information

Dataset: Diabetes 130-US Hospitals Dataset

* 101,766 patient records
* 50 original features
* 130 US hospitals
* Years: 1999–2008

Features include:

* Demographics
* Diagnosis codes
* Medication usage
* Laboratory results
* Hospital visit history
* Previous admissions and emergency visits

### Target Variable

`readmitted`

Binary Classification:

* 0 = Not readmitted within 30 days
* 1 = Readmitted within 30 days (<30)

---

## Data Preprocessing

The following preprocessing steps were applied:

* Replace missing values represented by "?"
* Remove duplicate patient encounters
* Drop weak and highly missing columns (such as weight)
* Handle missing categorical values (race, payer_code, medical_specialty)
* Remove invalid/deceased records
* Encode categorical variables
* Apply StandardScaler
* Feature selection using Random Forest importance
* PCA for dimensionality reduction
* Add 10% label noise to simulate adversarial real-world medical record errors

Final processed dataset:

* 69,970 rows
* PCA components after reduction

---

## Machine Learning Models

The following models were trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* Naive Bayes
* KNN
* SVM
* Gradient Boosting

### Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score
* AUC
* ROC Curve
* Confusion Matrix

Hyperparameter tuning was performed using:

`GridSearchCV`

Class imbalance was handled using:

`class_weight='balanced'`

---

## Final Best Model

### Logistic Regression

Final Results:

* Accuracy = 0.631
* AUC = 0.622
* Recall = 0.53
* F1-score = 0.21

Although some models achieved higher accuracy, they failed to detect readmitted patients.

For example, Gradient Boosting achieved high accuracy but Recall = 0.00.

In healthcare, Recall is more important because missing high-risk patients is dangerous.

Logistic Regression provided the best balance of:

* Clinical usefulness
* Minority-class detection
* Interpretability
* Stable performance

---

## Key Findings

* `number_inpatient` was the strongest predictor
* Previous hospitalizations strongly increased readmission risk
* Older patients showed higher readmission rates
* Accuracy alone is misleading in imbalanced datasets
* Label noise reduced model reliability

---

## Future Work

Future improvements may include:

* SMOTE for class balancing
* Threshold tuning
* Voting ensemble models
* SHAP explainability
* Real hospital deployment as a clinical decision support system

---

## Repository Contents

* Jupyter Notebook
* Streamlit Application
* Dataset files
* Technical Report
* Presentation Slides
* Research Poster
* Portfolio Website

---

## Live Demo

Streamlit App:
(https://cs4082-machinelearning-project-uwgzoj5cjk3dfs4lvwdnuw.streamlit.app/)

Portfolio Website:
([file:///Users/aseel/Desktop/ML.proj/portfolioF.html](https://aseelmoh.github.io/CS4082-MachineLearning-Project/portfolioF.html))

