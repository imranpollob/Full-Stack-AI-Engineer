# Day 4: Calculus for Machine Learning - Integrals and Optimization

Welcome to Day 4! Yesterday we learned about Derivatives, which measure the *rate of change* at a specific point. Today, we look at the other fundamental pillar of Calculus: **Integrals**.

After that, we will take the gradient descent algorithm we built yesterday and upgrade it to handle real-world, massive datasets using a technique called **Stochastic Gradient Descent (SGD)**.

## Integrals
If derivatives are about "change," integrals are about "accumulation." Geometrically, an integral computes the *total area* under a continuous curve on a graph.

In Machine Learning, we don't use integrals quite as often as derivatives, but they are absolutely essential for one specific area: **Probability Distributions**. If you have a continuous distribution of data (like height or age), the only way to calculate the probability of a specific range of values is by taking the integral of that distribution's curve!

We can solve integrals symbolically in Python exactly like we solved derivatives: using `sympy`.

```python
# day_ex1.py 
import sympy as sp

# Define the symbol 'x'
x = sp.Symbol('x')

# Define a function: f(x) = e^(-x)
f = sp.exp(-x)

# 1. Indefinite Integral (No boundaries)
indefinite_integral = sp.integrate(f, x)
print("Indefinite integral: ", indefinite_integral)
# Output: -exp(-x)

# 2. Definite Integral (Area under the curve from 0 to Infinity!)
definite_integral = sp.integrate(f, (x, 0, sp.oo))
print("Definite Integral: ", definite_integral)
# Output: 1 
# Note: Because the area under this curve perfectly equals 1, 
# this function is mathematically valid as a Probability Distribution!
```

## Advanced Optimization: SGD
Yesterday, our Gradient Descent algorithm looked at *every single data point* before taking a single step down the slope. This is called "Batch" gradient descent.

If your dataset is the Titanic dataset (800 rows), that takes milliseconds. If your dataset is every image on the internet (used to train ChatGPT), looking at every data point takes a month.

The solution is **Stochastic Gradient Descent (SGD)**. "Stochastic" means random. Instead of looking at the entire dataset, we pick *one single random data point* (or a tiny "mini-batch"), estimate the gradient based on just that point, and take our step!

It creates a "drunken" zigzag path down the slope, but it is astronomically faster. Almost all deep learning models today use variations of SGD (like the famous **Adam** optimizer).

Let's look at `day4_ex2.py` to see it in code.

```python
# day4_ex2.py
import numpy as np

# 1. Generate massive, messy synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # 100 random X coordinates
# We use a linear formula (y = 3X + 4), but add random "noise" so it's not a perfect line
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (a column of 1s) to the matrix
X_b = np.c_[np.ones((100, 1)), X]

# 2. SGD Implementation
def stochastic_gradient_descent(X, y, theta, learning_rate, n_epochs):
    m = len(y) # The number of data points (100)
    
    # An 'epoch' means we have taken 'm' random steps!
    for epoch in range(n_epochs):
        for i in range(m):
            # Pick ONE random data point from our dataset of 100!
            random_index = np.random.randint(m)
            
            # Grab just that one slice (xi and yi)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            
            # The exact same Calculus dot product from yesterday, but applied to ONE point
            gradients = 2 * xi.T @ (xi @ theta - yi)
            
            # Take the step
            theta -= learning_rate * gradients
            
    return theta

# Initialize random weights (theta) and Train!
theta = np.random.randn(2, 1)
learning_rate = 0.01

theta_opt = stochastic_gradient_descent(X_b, y, theta, learning_rate, n_epochs=50)

# Did our AI "learn" the secret formula behind our noisy data?
print("Optimized Parameters:", theta_opt)
# Ideally, theta_opt should be extremely close to [4.0, 3.0]!
```

## Wrapping Up Day 4
You've now mastered the two sides of Calculus required for AI engineering. You know how models optimize their weights using gradients, and how we estimate those gradients on massive datasets using SGD.

Tomorrow, on **Day 5: Probability Theory**, we revisit the concept of Probability Distributions. We'll leave definitive math behind and enter the world of statistics and uncertainty!
