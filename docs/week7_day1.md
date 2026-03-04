# Day 1: Introduction to Ensemble Learning

Welcome to Week 7! If you ask one person to guess the weight of an ox, they will probably be wrong. But if you ask 1,000 people to guess the weight and average their answers, the average will be shockingly close to the exact true weight.

This is the "Wisdom of the Crowds". In Machine Learning, we call this **Ensemble Learning**.

## Why Ensemble?
Every machine learning model has a weakness:
*   A **Decision Tree** has high Variance (it overfits easily).
*   A **Logistic Regression** has high Bias (it draws a straight line and misses curves).

If we train *both* of them, plus a **k-NN** model, and force them to vote on the final answer, we mathematically cancel out their individual weaknesses!

## The Three Main Architectures
1.  **Voting / Stacking:** You train 3 completely different models (e.g., Logistic, Tree, k-NN) and simply average their predictions.
2.  **Bagging (Bootstrap Aggregating):** You train 100 copies of the *same* model (usually decision trees) independently on random shuffles of your data.
3.  **Boosting:** You train 100 models in a sequence. Tree 1 makes predictions. Tree 2 looks at what Tree 1 got wrong, and tries to fix the mistakes. Tree 3 fixes Tree 2, etc.

## Hands-On: The Voting Classifier
Let's build a simple Voting Ensemble using Scikit-Learn. In `day7_ex.py` (located in the Day 1 folder), we train three weak algorithms and bind them together.

```python
# day7_ex.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

# 1. Initialize three completely different "weak" learners
log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()

# 2. Bind them together into a super-model!
ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', log_model),
        ('decision_tree', dt_model),
        ('knn', knn_model)
    ],
    voting='hard' # 'Hard' voting means majority rules (e.g., 2 out of 3 models say True)
)

# 3. Train the ensemble (This automatically trains all 3 nested models!)
ensemble_model.fit(X_train, y_train)

# 4. Predict
y_pred_ensemble = ensemble_model.predict(X_test)
```

## The Results
If we evaluate the accuracy of the models individually, we get:
*   Logistic Regression: `0.97`
*   Decision Tree: `0.93`
*   k-NN: `0.97`

But the Ensemble Model scores `0.97` to `1.00`! By smoothing out the wild guesses of the individual algorithms, the Ensemble provides a much more robust, reliable prediction on unseen testing data.

## Wrapping Up Day 1
Voting is great, but manually tracking 3 different algorithms is annoying. What if we could just tell Scikit-Learn to automatically spawn 500 decision trees at once?

Tomorrow on **Day 2: Bagging and Random Forests**, we will explore the most famous Bagging algorithm in the world!
