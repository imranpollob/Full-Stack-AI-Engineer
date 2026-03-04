# Day 6: Automated Hyperparameter Tuning

Welcome to Day 6. Today we put everything we've learned together. We are going to take complex algorithms and tune them using robust validation strategies mathematically.

We have used `GradientBoostingClassifier`, but today we introduce a very unique architecture: the **Support Vector Machine (SVM)**.

## Support Vector Machines (SVM)
The SVM doesn't use trees. It attempts to slice your data in half by drawing a mathematically optimal road (hyperplane) right through the middle, separating Class A from Class B. 

But what if Class A is trapped *inside a circle* of Class B? A straight road will fail!

SVMs solve this using the **Kernel Trick**. They mathematically throw the 2D data into the 3rd dimension! Suddenly, in 3D space, the algorithm easily slices a flat plane beneath the floating Class A dots, solving a non-linear problem with flat linear math!

## Tuning the SVM
Because the math of throwing data into higher dimensions is complex, you MUST tune an SVM using Grid Search. The two most vital parameters are:
*   `C`: The Regularization parameter (How much do we penalize the algorithm for drawing the road in the wrong spot?)
*   `kernel`: Should it use linear lines, or the multi-dimensional "RBF" trick?

## Hands-On Let's Optimize!
Look at `day6_ex.py`. We pit a Grid Searched `GradientBoostingClassifier` against a Randomized Searched `SVC`!

```python
# day6_ex.py
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# 1. We must tune the C penalty, and the dimension Kernel!
# We test 10 different 'C' values logarithmically spaced from 0.001 to 1000!
param_dist = {
    'C': np.logspace(-3, 3, 10),
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

# 2. We initialize the RandomizedSearch
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist, 
    n_iter=20,     # Out of 80 options, randomly test 20!
    scoring='accuracy',
    cv=5,          # Stratified 5-Fold Validation!
    n_jobs=-1
)

# 3. Optimize the Hyperplane!
random_search.fit(X_train, y_train)

# 4. View results!
print(f"Best Parameters: {random_search.best_params_}")
print(f"Test Accuracy: {random_search.best_score_:.4f}")
```

## Wrapping Up Day 6
You have now used `GridSearchCV` on Ensembles, and `RandomizedSearchCV` on advanced mathematical Kernels. 

Tomorrow is the finale of Week 8. On **Day 7: The Capstone Project**, we are going to build the ultimate ML Pipeline. We will load real business data, impute it, encode it, scale it, and Unleash a massive hyperparameter tournament inside a `RandomForestClassifier`.
