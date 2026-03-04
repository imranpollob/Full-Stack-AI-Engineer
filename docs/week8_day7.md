# Day 7: Optimization Capstone Project

Welcome to the end of Week 8! Today we do exactly what an Enterprise Machine Learning Engineer does in Production. 

We are going to build an end-to-end pipeline that handles Data Cleaning, Scaling, Imbalanced CV Evaluation, and advanced Hyperparameter Tuning in one glorious script.

We are predicting Customer Churn again using the `Telco-Customer-Churn` dataset.

## The Production Pipeline
Look at `day7_ex.py`. We have constructed the absolute optimal workflow.

### Phase 1: Munging and Engineering
Raw data is completely useless to an algorithm. We must clean it.
```python
# Handle missing financial values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# Encode String Categories to Math 
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Mathematically Scale large dollar amounts to stabilize algorithms
scaler = StandardScaler()
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])
```

### Phase 2: The Baseline Model
Always train a "stupid" baseline model first! If our hyperparameter tuning doesn't mathematically beat the baseline, our entire tuning strategy failed.

```python
# Train Baseline (Default Hyperparameters)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print(f"Initial Model Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")
```

### Phase 3: Unleash the Tuner!
We build our parameter grid, and launch the `RandomizedSearchCV` tournament targeting a massive 20 iterations using 5-Fold Stratified Validation!

```python
# The Sandbox
param_dist = {
    'n_estimators': np.arange(50, 200, 10),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

# The Tuning AI
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Launch!
random_search.fit(X_train, y_train)
```

### Phase 4: Production Evaluation
The Tournament is over! We print the final optimized parameters, and evaluate the algorithm against its `Baseline` model!
```python
best_model = random_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")
```

Because of our specific architectural tuning, the Tuned Model should mathematically outscore the Baseline on unseen Data. We have successfully optimized a production model!

## Wrapping Up Week 8!
Congratulations! You have officially conquered "Classical" tabular machine learning. You know exactly what hyperparameters are, how to tune them using Optuna and RandomizedSearchCV, and how to evaluate them using K-Fold stratification.

If you can build everything we discussed in Weeks 5-8... you are a highly-capable Data Scientist.

But tabulated spreadsheets are only $20\%$ of all human data. What about Images? What about Voice recordings? What about Natural Language? Trees, Regressions, and k-NN completely fail when faced with a 10-Megabyte photo. 

Next week, we graduate to **Week 9: Neural Networks & Deep Learning**. We introduce `TensorFlow` and `PyTorch`, the frameworks that power ChatGPT, self-driving cars, and modern AI. See you there!
