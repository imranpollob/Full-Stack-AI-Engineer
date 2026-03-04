# Day 4: Regularization Techniques for Model Optimization

Welcome to Day 4. Today we pause our Hyperparameter search and look inward at the mathematical equations driving our models.

A complex algorithmic formula will inherently learn to Overfit its training data. We introduced this concept during Week 5 regressions, but today we formalize the mathematics of **Regularization**. Let's review the two ultimate penalties.

## $L1$ and $L2$ Mathematical Penalties
Regularization adds a mathematical "penalty" to the Loss Function. 
If a Model tries to draw a bizarre, wiggly line that perfectly tags every single dot in the training set, it does so by creating massive multiplicative Weights (Coefficients). 

We use Calculus to literally punish the model for using large Weights!

### 1. Ridge Regression (L2)
Ridge Regression mathematically squares the Weights ($\beta^2$) and adds them directly to the Cost Function. 
*   **The Result:** The algorithm shrinks all of its weights down as close to zero as mathematically possible to avoid the penalty. It perfectly smoothens out jagged, overfitted lines into smooth curves.

### 2. Lasso Regression (L1)
Lasso Regression takes the absolute value ($|\beta|$) of the Weights and adds them to the Cost function.
*   **The Result:** Lasso is violent. If a feature is even slightly useless, Lasso mathematically crushes its coefficient perfectly to `0.0`. It acts as an automatic Feature-Selector!

## Hands-On Let's Regulate!
Look at `day4_ex.py` on the California Housing dataset. We import `LinearRegression`, `Ridge`, and `Lasso`, training them all side-by-side.

```python
# day4_ex.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 1. Train pure Linear Regression (No Penalty)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print(f"Linear Regression MSE: {mean_squared_error(y_test, lr_model.predict(X_test)):.4f}")
print("Coefficients:", lr_model.coef_)

# 2. Train Ridge Regression (L2 Penalty)
ridge_model = Ridge(alpha=0.1) # Alpha controls how heavy the punishment is!
ridge_model.fit(X_train, y_train)
print(f"Ridge Regression MSE: {mean_squared_error(y_test, ridge_model.predict(X_test)):.4f}")
print("Coefficients:", ridge_model.coef_)

# 3. Train Lasso Regression (L1 Penalty)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
print(f"Lasso Regression MSE: {mean_squared_error(y_test, lasso_model.predict(X_test)):.4f}")
print("Coefficients:", lasso_model.coef_)
```

When you print the `coef_` inside Lasso, you will notice that massive chunks of the array are literally `0.000`! It completely ignored half the dataset during training!

## Elastic Net
What if you want the smooth stability of `Ridge` but the automatic feature-selection of `Lasso`? 

Scikit-Learn provides `ElasticNet(alpha=0.1, l1_ratio=0.5)`. This literally combines both penalties simultaneously! It is currently the most robust form of linear architecture.

## Wrapping Up Day 4
You now understand how to optimize standard mathematical regressions. Tomorrow, on **Day 5: Cross-Validation**, we upgrade our `K-Fold` logic to properly evaluate datasets that are mathematically imbalanced!
