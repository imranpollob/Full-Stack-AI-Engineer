# Day 7: Ensemble Learning Project

Welcome to the end of Week 7! Today is the final showdown. We are going to load the real-world **Telco Customer Churn** dataset.

It is our job to predict which subscribers are about to cancel their TV/Internet service (Churn = 1) and which are going to stay (Churn = 0). It is highly imbalanced because most customers do not churn. 

We will preprocess the data, synthesize fake churners using `SMOTE` to fix the imbalance, and train the "Big Three" ensemble models side-by-side to declare a champion using the powerful `ROC-AUC` scoring metric!

## The Pipeline

Look at `day7_ex.py`. 

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. LOAD AND PREPROCESS DATA
df = pd.read_csv("Telco-Customer-Churn.csv")
# ... [Handling missing variables, Encoding categorical columns, and Scaling logic] ...

# 2. FIX IMBALANCED DATA
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 3. COMMENCE THE TOURNAMENT!

# Model 1: Random Forest (Bagging)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Model 2: XGBoost (Standard Boosting)
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)

# Model 3: LightGBM (Leaf-Wise Boosting)
lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train_resampled, y_train_resampled)

# 4. EVALUATE ROC-AUC PROBABILITIES
roc_auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
roc_auc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])

print(f"Random Forest ROC-AUC: {roc_auc_rf:.2f}")
print(f"XGBoost ROC-AUC: {roc_auc_xgb:.2f}")
print(f"LightGBM ROC-AUC: {roc_auc_lgb:.2f}")
```

*(Note: While algorithms will naturally trade blows natively, CatBoost / LightGBM almost always win out of the box until heavy GridSearchCV tuning is applied).*

## Wrapping Up Week 7!
Congratulations! You have officially utilized the most bleeding-edge mathematical tabular algorithms in the world. You understand how Bagging fixes Variance by splitting trees, and Boosting fixes Bias by forcing models to calculate Residual Errors.

But wait. What if we have a model that takes 6 hours to train? We can't use `GridSearchCV` on it! Grid Searching across 200 parameters would take a full year!

Next week, we graduate to **Week 8: Model Tuning & Optimization**. We will learn how to use advanced Bayesian Statistics (`RandomizedSearchCV`, `Optuna`) to intelligently guess Hyperparameters, drastically speeding up training while squeezing out maximum performance! See you there!
