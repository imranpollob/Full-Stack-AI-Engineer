# Day 2: Grid Search and Random Search

Welcome to Day 2. Yesterday, we realized that guessing the perfect Hyperparameters by hand is impossible. Today, we automate the testing.

## 1. Grid Search (Exhaustive Search)
Grid Search is brute force. We hand it a dictionary of numbers, and it will methodically run a `for` loop, testing **every single possible combination** of hyperparameters. 

```python
from sklearn.model_selection import GridSearchCV
# ... [Load Dataset] ...

param_grid = {
    'n_estimators': [50, 100, 150],       # 3 options
    'max_depth': [None, 5, 10],           # 3 options
    'min_samples_split': [2, 5, 10]       # 3 options
}

# 3 * 3 * 3 = 27 totally unique models!
# With cv=5, it actually trains 135 models!
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, n_jobs=-1
)
grid_search.fit(X_train, y_train)
```
*   **Pros:** It is guaranteed to find the absolute best combination inside your dictionary.
*   **Cons:** If your dictionary has 1,000 combinations, Grid Search might take 3 days to finish running!

## 2. Random Search
What if we don't have 3 days? What if we have 5 minutes? 

**Random Search** accepts a massive distribution of numbers. Instead of testing all of them, you tell it `n_iter=20`. It will blindly pick 20 random mathematical combinations, test them, and give you the best one.

```python
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# We pass massive lists!
param_dist = {
    'n_estimators': np.arange(50, 200, 10), # [50, 60, 70... 190]
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10, 20]
}

# There are hundreds of combinations, but we only test 20!
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20, # Only test 20 random models!
    cv=5, n_jobs=-1
)
random_search.fit(X_train, y_train)
```

## The Showdown
In `dat2_ex.py`, we ran both sequentially! 
*   **Grid Search** meticulously found the best answer constrained by its tiny block of limits.
*   **Random Search** found a highly-competitive answer in a fraction of the time, often discovering numbers the Grid Search wasn't even allowed to look at!

## Wrapping Up Day 2
If your dataset is small and your model trains fast: use `GridSearchCV`.
If your dataset has millions of rows and takes 1 hour to train: use `RandomizedSearchCV`.

But Random Search is... random. It doesn't use logic. What if the Search algorithm was an AI itself? What if it *learned* from its guesses? 

Tomorrow, in **Day 3: Bayesian Optimization**, we unlock the ultimate hyperparameter tuning strategy!
