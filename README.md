# 🧠 Machine Learning Pipeline Project

## 📄 Overview
This project demonstrates a **complete end-to-end machine learning pipeline** — from data preprocessing and feature engineering to model training, evaluation, and interpretability — all wrapped in a single reproducible Python script.

The goal is to build a flexible, automated framework that can handle both **regression** and **classification** tasks with minimal modification.  
For demonstration, this pipeline was tested on the **House Prices Dataset (`train.csv`)** from Kaggle.

---

## ⚙️ Features
- ✅ Automated data loading and cleaning  
- ✅ Smart handling of missing values (mean / most frequent imputation)  
- ✅ Encoding of categorical variables using `OneHotEncoder`  
- ✅ Feature scaling using `StandardScaler`  
- ✅ Automatic detection of regression or classification problem  
- ✅ Model training using `RandomForest` (can be swapped with XGBoost / Linear models)  
- ✅ Cross-validation performance evaluation  
- ✅ Feature importance visualization  
- ✅ SHAP explainability plots  
- ✅ Model export via `joblib`  

---

## 🧩 Tech Stack
- **Language:** Python 3.9+
- **Libraries:**
  - pandas  
  - numpy  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - shap  
  - joblib  

---

## 📊 Pipeline Architecture
┌──────────────────────────┐
│ Load Data │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Split Train / Test │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Preprocess Features │
│ (Impute, Encode, Scale) │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Train Model (RF) │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Evaluate Performance │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Feature Importances │
│ & SHAP Interpretability │
└────────────┬─────────────┘
▼
┌──────────────────────────┐
│ Save Final Model │
└──────────────────────────┘

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Psychaostics/ml-pipeline-project.git
cd ml-pipeline-project
```
### 2️⃣ Install dependencies

