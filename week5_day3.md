# Day 3: Advanced Regression Models - Polynomials and Regularization

Welcome to Day 3. So far, we've used Linear Regression. But what happens if you try to draw a straight line through data that goes in a circle? 

If your model is too simple for the data, you get **Underfitting**. The model is completely mathematically blind to complex patterns. To fix this, we use **Polynomial Regression**.

## Polynomial Regression
Polynomial transformations physically alter the input data before it hits the regression algorithm. If you feed the model $X$, it can only draw a straight line ($y = 3x$). 

But if you use `PolynomialFeatures` to feed the model $[x, x^2, x^3]$, the algorithm will gain the ability to draw mathematical curves ($y = 3x^2 + 2x$)!

Let's look at `day3_samples2.py` as an example:

```python
# day3_samples2.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate data in a U-Shape curve (y = 3x^2)
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5

# Transform the features! Now our AI can see X squared!
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# We use the exact same Linear Regression model as yesterday
model = LinearRegression()
model.fit(X_poly, y)
```

## The Danger of Overfitting
If you transform your data to `degree=100`, the AI will draw a wildly jagged curve that perfectly tags every single dot in the training set.

This is **Overfitting**. The AI has lost the plot. It isn't finding the trend; it's just memorizing the noise.

## Regularization to the Rescue (Ridge & Lasso)
To stop Overfitting, we use **Regularization**. This technique adds mathematical "penalties" to the model's loss function. If the model tries to draw crazy jagged lines by making its internal weights ($\beta$) massive, the penalty triggers and increases the Error!

There are two major types, and Scikit-Learn implements both out of the box:
1.  **Ridge Regression (L2):** Shrinks the AI weights down close to zero, forcing the line to be smoother.
2.  **Lasso Regression (L1):** Highly aggressive! It shrinks the weights of "useless" features exactly *to zero*, completely deleting them from the equation. It acts as an automatic feature-selector!

Let's look at `day3_ex1.py` to see them applied to the California Housing Dataset:

```python
# day3_ex1.py
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# ... (Data loading and polynomial transformation code hidden for brevity)

# Train a Ridge Regression Model (alpha controls how strong the penalty is)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Train a Lasso Regression Model 
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Evaluate!
print("Ridge Regression MSE:", mean_squared_error(y_test, ridge_predictions))
print("Lasso Regression MSE:", mean_squared_error(y_test, lasso_predictions))
```

## Wrapping Up Day 3
If your AI is underfitting, add complexity using Polynomials. If your AI is overfitting, rein it in using Ridge or Lasso Regularization!

Everything we have done so far predicts *numbers*. Tomorrow on **Day 4: Introduction to Classification**, we fundamentally shift gears. How do we predict categories? We use Logistic Regression.
