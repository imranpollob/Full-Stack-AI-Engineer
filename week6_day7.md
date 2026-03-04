# Day 7: Cross Validation and Hyperparameter Tuning

Welcome to the end of Week 6. You now know the complete pipeline:
`Data -> Fill Missing -> Scale -> Encode -> Split -> Train -> Evaluate`.

Today, we learn how to do all of that with two lines of code using `ColumnTransformer` and build a model that recursively tunes *itself* using `GridSearchCV`.

## 1. The ColumnTransformer
Manually `fit_transform`ing six different pandas columns is tedious. Scikit-Learn provides `ColumnTransformer` to automate the entire Feature Engineering process in one swoop!

```python
# day7_project.py (Loading Titanic Survival Dataset)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ... Clean Missing Values First ... 

# We build an array of instruction tuples! 
# We tell the Transformer EXACTLY what to do with what columns.
preprocessor = ColumnTransformer(
    transformers=[
        # Apply the StandardScaler ONLY to 'Age' and 'Fare'
        ('num', StandardScaler(), ['Age', 'Fare']),
        # Apply the OneHotEncoder ONLY to categorical strings
        ('cat', OneHotEncoder(), ['Pclass', 'Sex', 'Embarked'])
    ]
)

# Boom! The entire dataset is scaled, encoded, and transformed in one command!
X_preprocessed = preprocessor.fit_transform(X)
```

## 2. Hyperparameter Tuning using GridSearch
Every model has internal settings you must pick. Let's look at `RandomForestClassifier`. How many trees should be in the forest (`n_estimators`)? How deep should each tree be (`max_depth`)?

These are **Hyperparameters**.

Instead of manually changing the code and re-running the script 50 times, we can use `GridSearchCV`. We give it a "Grid" (Dictionary) of all the settings we want to try. It will use a `for` loop to systematically train a new model on *every possible combination of settings*, use K-Fold Cross Validation to ensure they are robust, and output the absolute highest-scoring set of rules!

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. Define the parameters you want to test!
# (3 estimators * 3 depths * 3 splits = 27 totally different models!)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 2. Set up the GridSearch autonomous bot 
# (It will test all 27 models 5 independent times via cv=5... that's 135 models!)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,       # Use 5-Fold Cross Validation for safety!
    n_jobs=-1   # Use all cores of your computer CPU simultaneously!
)

# 3. Hit run, and watch your laptop fan spin up!
grid_search.fit(X_preprocessed, y)

# 4. Which of the 27 combinations won the tournament?
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.2f}")

# Output: Best hyperparameters: {'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 50}
# With an Accuracy of 83% Survival Rate prediction!
```

## Wrapping Up Week 6!
Congratulations! You just built a professional, production-grade Machine Learning pipeline. You engineered a beautiful dataset, and dynamically grid-searched your way to the pinnacle of Accuracy.

Next week, we graduate to **Week 7: Advanced Machine Learning Algorithms**. We will look at Ensemble models, Boosting, and the most dominant tabular algorithm in the world: `XGBoost`. See you there!
