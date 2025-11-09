"""
House Price Prediction Project
------------------------------
Goal: Build an end-to-end predictive model using RandomForest
Steps: Data Cleaning → Preprocessing → Modeling → Evaluation → Interpretability
Dataset: House Prices (Kaggle)
Tools: Python, pandas, scikit-learn, SHAP, seaborn, matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import shap
import os

# Create directories for visuals and models
os.makedirs('visuals', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 2. Load dataset (replace path)
df = pd.read_csv("U:\\New folder\\Python\\Projects\\Datasets\\train.csv")
print("Shape:", df.shape)
print(df.head())
print("✅ Step 1: Data loaded")
# 3. Quick EDA
print(df.info())
print(df.describe(include='all').T)
print("Missing per column:\n", df.isnull().sum().sort_values(ascending=False).head(20))

# remove duplicates
df = df.drop_duplicates()

# 4. Target identification
target = 'SalePrice'  # <-- change to actual target column
y = df[target]
X = df.drop(columns=[target])

# 6. Feature types
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric:", num_cols)
print("Categorical:", cat_cols)
# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42,
                                                    stratify=y if y.nunique() <= 10 else None)

# 8. Preprocessing pipelines
# Numerical pipeline: Imputation and Scaling
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline: Imputation and One-Hot Encoding
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')
print("✅ Step 2: Preprocessing complete")
# 9. Choose model based on task type
is_regression = True if y.dtype in [np.float64, np.int64] and y.nunique() > 10 else False

if is_regression:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    scoring = 'neg_root_mean_squared_error'
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    scoring = 'roc_auc'  # or 'accuracy' depending on class balance
# 10. Build full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# 11. Quick baseline CV
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scoring)
print("Baseline CV scores:", cv_scores)
print("CV score mean:", np.mean(cv_scores))

# 12. Fit pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("✅ Step 3: Model training done")
# 13. Evaluation
if is_regression:
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.3f}, R2: {r2:.3f}")
else:
    # for classifiers get predicted probabilities and metrics
    y_prob = pipeline.predict_proba(X_test)[:,1]
    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except:
        pass
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("✅ Step 4: Evaluation done")
# 14. Feature importance (RandomForest / XGBoost)
# Get preprocessor output column names
def get_feature_names(preprocessor):
    feature_names = []
    # numeric names
    if num_cols:
        feature_names.extend(num_cols)
    # onehot names
    if cat_cols:
        ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
        # Use get_feature_names_out() without input_features
        cat_ohe_cols = ohe.get_feature_names_out().tolist()
        feature_names.extend(cat_ohe_cols)
    return feature_names

feat_names = get_feature_names(pipeline.named_steps['preprocessor'])
model_step = pipeline.named_steps['model']

importances = None
if hasattr(model_step, 'feature_importances_'):
    importances = model_step.feature_importances_
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
    plt.figure(figsize=(8,6))
    sns.barplot(x=fi.values, y=fi.index)
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('visuals/feature_importances.png', dpi=150)
    plt.show()

# 15. SHAP (model interpretability)
X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train).toarray()
X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test.sample(200, random_state=42)).toarray()

explainer = shap.Explainer(model_step, X_train_transformed)
shap_values = explainer(pd.DataFrame(X_test_transformed, columns=feat_names), check_additivity=False)
shap.plots.beeswarm(shap_values, show=True)

# 16. Save model
joblib.dump(pipeline, 'models/final_pipeline.pkl')

# 17. Quick conclusions cell (write your insights)
print("✅ Model training & evaluation complete. Save visuals in /visuals and model in /models.")