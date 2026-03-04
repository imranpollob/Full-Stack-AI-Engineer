# Day 7: Supervised Learning Mini Project

Welcome to the end of Week 5! You now know how to train, predict, and evaluate Machine Learning models. 

Today we simulate a real-world task. A telecom company needs you to predict Customer Churn (which customers are going to cancel their subscription). You will load a real dataset, clean it, train *both* Logistic Regression and k-NN, and prove which one performs better.

## Task 1: EDA and Preprocessing
The dataset contains a lot of categorical variables (`gender`, `payment_method`, `contract_type`). Machine Learning models only speak math. We must encode these words into numbers using Scikit-Learn's `LabelEncoder`.

Because we plan on training a Distance-based algorithm (k-NN), we must also scale all the data using `StandardScaler`.

```python
# mini_project_day7_telco.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df_telco = pd.read_csv('Telco-Customer-Churn.csv')

# 1. ENCODE WORDS TO NUMBERS
le = LabelEncoder()
df_telco['churn'] = le.fit_transform(df_telco['churn'])
df_telco['gender'] = le.fit_transform(df_telco['gender'])
df_telco['contract_type'] = le.fit_transform(df_telco['contract_type'])
df_telco['payment_method'] = le.fit_transform(df_telco['payment_method'])

X = df_telco.drop(columns=['churn'])
y = df_telco['churn']

# 2. SCALE THE DATA (For k-NN)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Task 2: The Showdown
Let's initialize our two rival models and train them on the exact same `X_train` dataset!

```python
# Train logistic regression model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Train k-NN model (We'll use k=5)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Have them both take the final exam
log_pred = log_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
```

## Task 3: Evaluating the Winner
We don't just look at Accuracy. A telecom company cares about **Recall**: They want to catch *every* single customer who is going to churn so they can send them a discount code. 

```python
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_pred))

print("\nk-NN Classification Report:")
print(classification_report(y_test, knn_pred))
```
*Depending on the specifics of the data, one model will usually outperform the other in Recall or F1-Score!*

Take a look at the `confusion_matrix` to see exactly how many False Positives and False Negatives each algorithm generated.

## Wrapping Up Week 5!
Congratulations! You are officially doing Machine Learning. You have built a complete, end-to-end pipeline that takes raw business data and outputs statistically evaluated predictions.

But did you notice something annoying? Encoding strings into numbers manually was tedious. And what if some missing data completely breaks the model? 

Next week, we enter **Week 6: Feature Engineering**. We will learn how to professionally handle missing data, advanced categorical encoding (One-Hot), and automated feature selection pipelines! See you there.
