# Day 3: Boosting and Gradient Boosting

Welcome to Day 3. So far, we've used Random Forests. That strategy builds 100 massive trees completely independently of one another.

Today, we introduce **Boosting**. Boosting trains very tiny, "weak" trees one at a time. 
*   **Tree 1** makes predictions. It will get some right, and some wrong.
*   **Tree 2** looks at the data that Tree 1 got wrong, and mathematically forces itself to focus *only* on fixing those specific mistakes.
*   **Tree 3** focuses on the mistakes of Tree 2.

We repeat this 100 times. What is left is a highly-tuned sequence of models that mathematically annihilate errors!

## Gradient Boosting
The most reliable boosting framework relies on the Calculus we learned in Week 3: **Gradient Boosting**. The trees literally utilize Gradient Descent to calculate the mathematical "Residuals" (errors) of the previous tree, stepping closer to perfect accuracy with every iteration.

Let's look at `day3_ex.py` to compare a `RandomForest` against a `GradientBoostingClassifier` on the Breast Cancer dataset.

```python
# day3_ex.py
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Train Gradient Boosting (Sequential Learning)
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb)}")

# 2. Train Random Forest (Independent Bagging)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
```

*Depending on the complexity of the dataset, Gradient Boosting will usually beat Random Forest in raw accuracy!*

## Tuning the Boost
Gradient Boosting is powerful, but it is highly prone to Overfitting. Because it is actively trying to hunt down and fix every single tiny error, it will completely memorize the training noise if you let it run for too long!

We control this with crucial hyperparameters:
*   `learning_rate`: How aggressively should Tree 2 fix Tree 1? (Lower is safer, `0.01` to `0.2`).
*   `n_estimators`: How many sequential trees should we build in the chain?
*   `max_depth`: Unlike Random Forests that use massive depths (10-20), Boosting usually relies on very shallow "stubs" (Depth of 3 to 7).

```python
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}
```

## Wrapping Up Day 3
Bagging (Random Forest) is safe, fast (because trees train in parallel), and rarely overfits. 
Boosting (Gradient Boosting) is slower (because trees train sequentially), requires careful tuning to avoid overfitting, but almost always wins in Kaggle competitions.

But Scikit-Learn's implementation of Gradient Boosting is notoriously slow. Tomorrow, on **Day 4: XGBoost**, we install a third-party library that optimized this algorithm to such an extreme degree it changed the Data Science industry forever.
