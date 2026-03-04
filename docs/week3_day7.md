# Day 7: Mini-Project - Linear Regression from Scratch!

Welcome to the end of Week 3. Today, we pull together everything we've learned—Linear Algebra, Calculus, and Statistics—and build an actual predictive Machine Learning model from scratch.

We will not use Scikit-Learn. We are going to build the algorithms that power Scikit-Learn.
We are building a **Linear Regression** model.

## The Mathematical Process
Let's understand the math before we write the code:
1. **The Model (Linear Algebra):** We predict our values ($\hat{y}$) using the equation $\hat{y} = X \cdot \theta$. We take our feature matrix $X$ and perform a dot product with our weight parameters $\theta$.
2. **The Loss (Statistics):** How wrong is our model? We calculate the Mean Squared Error (MSE). We want this cost function $J(\theta)$ to be as small as possible.
3. **The Optimization (Calculus):** How do we minimize the MSE? We calculate the partial derivatives (the Gradient) of the cost function with respect to our weights! We then use Gradient Descent to update $\theta$ and fall down the slope!

## Hands-On Let's Code!
Let's look at `mini_project.py`. We generate some synthetic data, and then we build three distinct modules:

```python
# mini_project.py
import numpy as np

# A. Generate Synthetic data (Y = 3X + 4 + noise)
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term to feature matrix (The y-intercept!)
X_b = np.c_[np.ones((100, 1)), X]

# Initialze weights randomly
theta = np.random.randn(2, 1)
learning_rate = 0.1
iterations = 1000

# --------------------------
# Task 1: The Model (Linear Algebra)
# --------------------------
def predict(X, theta):
    return np.dot(X, theta)

# --------------------------
# Task 2: Optimization (Calculus Gradient Descent)
# --------------------------
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        # 1. Calc Predictions | 2. Calc Error | 3. Calc Gradient via Vector Math!
        gradients = (1/m) * np.dot(X.T, (np.dot(X, theta)- y))
        
        # 4. Take the downhill step
        theta -= learning_rate * gradients
    return theta

# --------------------------
# Task 3: The Evaluation (Statistics)
# --------------------------
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r_squared(y_true, y_pred):
    # Calculates how much Variance in Y is actually caused by X!
    ss_res = np.sum((y_true - y_pred)** 2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)  

# ==========================
# Run the Pipeline!
# ==========================

# Perform gradient descent
theta_optimized = gradient_descent(X_b, y, theta, learning_rate, iterations)

# Predictions and evaluations using our optimized weights
y_pred = predict(X_b, theta_optimized)
mse = mean_squared_error(y, y_pred)
r2 = r_squared(y, y_pred)

print("Optimized Parameters (Theta): \n", theta_optimized)
# Output should be very close to our formula: 4 (Y-Intercept) and 3 (Slope)!

print("MSE: ", mse)
print("R2: ", r2) 
# An R2 close to 1.0 means our model perfectly predicts the data!
```

## Wrapping Up Week 3!
Congratulations. You didn't just type `fit()` and `predict()` using a black box library. You wrote the algebraic dot products, calculated the calculus gradients, and evaluated the statistical variance manually. 

You have just built your very first Machine Learning algorithm from the ground up!

**Next Week**: In **Week 4**, we build on these math concepts, focusing heavily on Probability and Statistics to perform feature engineering, hypothesis tests, and deeper analysis before deploying these models! See you then!
