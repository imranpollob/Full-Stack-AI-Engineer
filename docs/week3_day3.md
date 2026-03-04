# Day 3: Calculus for Machine Learning - Derivatives

Welcome to Day 3. Linear algebra gives our data its shape. Calculus is how our AI models actually learn. 

When you train a neural network, you are mathematically trying to find the point where its "Error" (or Loss) is the absolute lowest. How do we find the bottom of this mathematical valley? We use **Derivatives**.

## What are Derivatives?
In pure math, a derivative measures how fast a function is changing at a specific point. If the function is a curve, the derivative gives you the *slope* of the curve at any given point.

If the slope is pointing down, the AI knows it needs to move in that direction to minimize its error!

Normally, calculating derivatives is tedious. In Python, we can use the `sympy` module to do symbolic math perfectly.

```python
# day3_ex1.py
import sympy as sp

# Define the symbol 'x'
x = sp.Symbol('x')

# Define a function: f(x) = x^3 - 5x + 7
f = x**3 - 5*x + 7

# Calculate the algebraic Derivative instantly!
derivative = sp.diff(f, x)

print("Function: ", f)
print("Derivative: ", derivative)
# Output: Derivative: 3*x**2 - 5
```

## Partial Derivatives and Gradients
Real AI models don't just have one variable (`x`). They have millions of variables (weights). To find the slope of a 3D (or million-D) surface, we must calculate the derivative in *every single direction*. These are called **Partial Derivatives**.

When you combine all the partial derivatives into a single Vector, that vector is called a **Gradient**.

```python
# day3_ex2.py
import sympy as sp

# Define a multivariable function: f(x,y) = x^2 + 3y^2 - 4xy
x, y = sp.symbols('x y')
f = x**2 + 3*y**2 - 4*x*y

# Compute partial derivatives
grad_x = sp.diff(f, x) # The slope along the X axis
grad_y = sp.diff(f, y) # The slope along the Y axis

print("Gradients:")
print("Grad X:", grad_x)
print("Grad Y:", grad_y)
```

## Gradient Descent Parameter Optimization
The holy grail. **Gradient Descent** is the algorithm used to train almost every modern neural network.

The algorithm is beautifully simple:
1.  Initialize parameters (weights) randomly.
2.  Calculate the Gradient (slope) of the current loss.
3.  Update the parameters in the *opposite* direction of the gradient.
4.  Repeat a few thousand times.

Let's look at `day3_ex3.py` to see this implemented from scratch in pure NumPy!

```python
# day3_ex3.py
import numpy as np

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y) # m = Number of training examples
    
    for _ in range(iterations):
        # 1. Make a prediction using our current parameters (theta)
        predictions = np.dot(X, theta)
        
        # 2. Calculate our Error
        errors = predictions - y
        
        # 3. Calculate the Gradients via Matrix Dot Product calculus
        gradients = (1/m) * np.dot(X.T, errors)
        
        # 4. Update our theta parameters by stepping "down" the slope!
        theta -= learning_rate * gradients
        
    return theta

# Sample Data & Hyperparameters
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([2, 2.5, 3.5])
theta = np.array([0.1, 0.1])
learning_rate = 0.1 # This is how large of a 'step' we take downhill
iterations = 1000

optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)
print("Optimized Parameters: ", optimized_theta)
```

## Wrapping Up Day 3
Gradient Descent is the very heartbeat of machine learning. You have now written the fundamental algorithm that makes ChatGPT "smart."

But this algorithm has a weakness: if your dataset has a billion rows, calculating the exact gradient at every step is too slow.

Tomorrow, on **Day 4: Calculus for Machine Learning (Integrals and Optimization)**, we will fix this problem by introducing **Stochastic Gradient Descent (SGD)**!
