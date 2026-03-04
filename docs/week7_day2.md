# Day 2: Bagging and Random Forests

Welcome to Day 2. Yesterday we looked at a Voting Classifier that merged three different algorithms. Today, we focus on a technique where we use *one* algorithm, but spawn hundreds of clones of it: **Bagging**.

## What is Bagging? (Bootstrap Aggregating)
If you train 100 Decision Trees on the exact same dataset, they will all draw the exact same tree. Voting won't help if everyone votes the exact same way.

To fix this, **Bagging** takes your dataset and creates 100 random, slightly-mutated *subsets* of your data using a statistical trick called "Bootstrapping" (random sampling *with replacement*).

It trains one tree on Subset A, one tree on Subset B, etc. Now, you have 100 unique "expert" trees that each learned slightly different patterns in the data!

## The Random Forest
When you apply Bagging specifically to Decision Trees, you create a **Random Forest**.

The Random Forest goes one step further than standard Bagging. Not only does it mathematically shuffle the *Rows* (Bootstrapping), but it also randomly hides *Columns* (Features) from every tree! 
If one tree isn't allowed to see the "Age" column, it is forced to discover hidden correlations in the other columns. This creates massive diversity in our army of trees.

```python
# day2_ex.py
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest
# random_state ensures reproducibility. 
# By default, Scikit-Learn spawns 100 trees inside this forest!
rf_model = RandomForestClassifier(random_state=42)

# Train all 100 trees natively!
rf_model.fit(X_train, y_train)

# When we predict, all 100 trees vote on the answer!
y_pred = rf_model.predict(X_test)
```

## Tuning the Forest (GridSearch)
A Random Forest has critical hyperparameters that we must tune to maximize accuracy:
*   `n_estimators`: How many trees in the forest? (Too few = weak. Too many = slow).
*   `max_depth`: How deep can each tree grow? (Prevents overfitting).
*   `max_features`: How many columns should be hidden from each tree? 

Let's use `GridSearchCV` to autonomously find the best layout for our forest!

```python
from sklearn.model_selection import GridSearchCV

# Define the matrix of parameters we want the AI to test
param_grid = {
    'n_estimators': [50, 100, 200],              # Test 50, 100, and 200 trees
    'max_depth': [None, 10, 20],                 # Test infinite depth vs restricted
    'max_features': ['sqrt', 'log2', 'None']  # How many columns to randomly use
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,               # Use 5-Fold Cross Validation
    scoring='accuracy',
    n_jobs=-1           # Use all CPU cores!
)

grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
```

## Wrapping Up Day 2
Random Forests are famously robust. Because they average out hundreds of trees, it is almost impossible for a Random Forest to Overfit. It is usually the best "Baseline" model any data scientist will start with.

But what if we want to be aggressive? Instead of training 100 independent trees and averaging them out, what if we train the trees *sequentially* so they can learn from each other's mistakes?

Tomorrow on **Day 3: Boosting and Gradient Boosting**, we introduce the alternative to Bagging!
