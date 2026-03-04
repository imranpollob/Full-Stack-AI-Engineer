# Day 2: Supervised Learning and Regression Models

Welcome to Day 2! Today we train our first official Machine Learning algorithm.

**Supervised Learning** is the process of training a model on *labeled* data. It falls into two major categories:
1.  **Classification:** Predicting discrete categories (e.g., Is this email Spam or Not Spam?).
2.  **Regression:** Predicting continuous numbers (e.g., What will the temperature be tomorrow?).

Today, we focus on Regression.

## Linear Regression Recap
In Week 4, we showed how Scikit-Learn can fit a mathematical line through a dataset to minimize the Mean Squared Error (MSE). We used it to calculate statistical correlations. 

Today, we take that exact same math and use it to predict the future.

The workflow for almost every Supervised algorithm in Scikit-Learn follows these exact steps:
1.  **Split** the data into Training and Testing sets.
2.  **Fit** (`model.fit()`) the model on the *Training* set, allowing it to mathematically discover the weights/slope using Gradient Descent.
3.  **Predict** (`model.predict()`) the target answers for the *Testing* set!

## Hands-On Let's Code!
Let's look at `day2_ex1.py`. We generate 100 random data points representing a mysterious relationship. We want the AI to learn that relationship.

```python
# day2_ex1.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Generate synthetic data (A messy, noisy line where y is roughly 3*X)
np.random.seed(42)
X = np.random.rand(100, 1) * 100
y = 3 * X + np.random.randn(100, 1) * 2

# 2. Split Data (Hide 20% from the model!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Linear Regression ONLY on the Training Data
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions on the hidden Testing Data
y_pred = model.predict(X_test)
```

### Evaluating the Model
Because `LinearRegression` is an algebraic model, we can actually look inside its brain to see what it learned:

```python
print("Slope : ", model.coef_[0][0])
print("Intercept : ", model.intercept_[0])
# Output: Slope: ~3.00, Intercept: ~0.16. 
# The model successfully figured out the hidden multiplier in our synthetic data!
```

To scientifically prove our model is good, we compare its `y_pred` guesses against the actual `y_test` hidden answers.

```python
# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse) # Output: ~3.67
print("R-Squared: ", r2) # Output: ~0.999!
```
An $R^2$ of `0.999` tells us our model successfully mapped 99.9% of the mathematical variance in our testing set!

## Wrapping Up Day 2
Congratulations! You have officially trained a Supervised Machine Learning algorithm. You gave it raw data, it learned the underlying mathematical relationship using gradient descent, and it successfully proved its intelligence by accurately predicting the hidden testing set.

But what if the relationship in the data isn't a straight line? What if it's a curve? 
Tomorrow on **Day 3: Advanced Regression Models**, we learn how to mathematically bend our predictions using Polynomial Transformations!
