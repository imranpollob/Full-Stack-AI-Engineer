# Day 4: Introduction to XGBoost

Welcome to Day 4. Yesterday we looked at Scikit-Learn's `GradientBoostingClassifier`. It is powerful, but it has a massive problem: Speed. 

Because Boosting physically requires Tree 2 to wait for Tree 1 to finish training so it can analyze the errors, it cannot be parallelized across your CPU cores. It is mathematically forced to run on a single thread.

In 2014, researchers released **eXtreme Gradient Boosting (XGBoost)**. It fundamentally solved the speed problem by mathematically approximating splits using Histograms, and cleverly parallelizing the construction of the *nodes* inside each tree.

XGBoost is so universally dominant that it has won almost every single Kaggle tabular/structured data competition for a decade.

## Why XGBoost?
Beyond speed, XGBoost introduced massive mathematical improvements to standard boosting:
1.  **Native Regularization:** XGBoost natively calculates $L1$ and $L2$ Regularization penalties (just like Ridge and Lasso) during its Tree splits, completely obliterating Overfitting out of the box.
2.  **Native Missing Values:** If your dataset has missing data (`NaN`), XGBoost doesn't care! It automatically learns which direction to send missing values down the tree splits. 
3.  **DMatrix:** It utilizes an advanced, cache-optimized C++ matrix format physically loading data into RAM differently than Pandas, boosting compute speed.

## Hands-On Let's Train!
We can use the native C++ API (`xgb.train`), or we can use the convenient Scikit-Learn wrapper (`XGBClassifier`) which allows us to use `GridSearchCV` just like Random Forests!

Look at `day4_ex.py` on the Breast Cancer dataset:

```python
# day4_ex.py
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 1. Initialize the XGBoost Classifier
xgb_clf = XGBClassifier(
    eval_metric='logloss', # What math formula should tell the Trees they are wrong?
    random_state=42
)

# 2. Let's Tune the Beast!
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],   # Step size towards the gradient
    'n_estimators': [50, 100, 200],      # How many boosting rounds (trees)?
    'max_depth': [3, 5, 7],              # Tree depth
    'subsample': [0.8, 1.0],             # Bagging randomly select 80% rows
    'colsample_bytree': [0.8, 1.0]       # Bagging randomly select 80% columns
}

# 3. Fire the GPU/CPU Grid Search!
grid_search = GridSearchCV(
    estimator=xgb_clf, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Output: Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.1, 
# 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}
```

## Wrapping Up Day 4
You now have access to the actual algorithm used in Enterprise Financial, Medical, and Retail prediction tasks. It handles missing data gracefully, fights overfitting beautifully, and trains efficiently.

But... it still has one weakness. We still have to meticulously `OneHotEncode` all of our Text columns using Pandas before training!

Tomorrow on **Day 5: LightGBM and CatBoost**, we look at two massive competitors (from Microsoft and Yandex) that solved the Text problem forever.
