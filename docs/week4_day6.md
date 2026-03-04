# Day 6: Correlation and Regression Analysis

Welcome to Day 6! Tomorrow, you build your final project for Math Week. Before you can build it, you must select the *right* data to feed it. 

If you throw 5,000 columns of data into an AI model, it will take hours to train and probably overfit. You must filter the data down to only the columns that mathematically matter. How? **Correlation**.

## Understanding Correlation
Correlation measures the strength and direction of the linear relationship between two continuous variables. 
*   **1.0:** Perfect positive relationship (As X goes up, Y goes exactly up).
*   **-1.0:** Perfect negative relationship (As X goes up, Y goes exactly down).
*   **0.0:** No relationship. The variables are completely independent.

In Week 2 we used Pandas `.corr()` and Seaborn `heatmap` visually. Let's look at `day6_ex1.py` to see the math again using the famous Iris dataset:

```python
# day6_ex1.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# We drop the categorical species column before doing math
del df["species"]

# This calculates the Pearson Correlation Coefficient (r) for every pair!
correlation_matrix = df.corr()

# Visualize the resulting 1.0 to -1.0 scores!
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Feature Correlations")
plt.show()
```
*If two different input features (like `petal_length` and `petal_width`) have a 0.96 correlation, it means they contain almost the exact same information. You should probably delete one before training an AI!*

## Linear Regression (Using Scikit-Learn)
Last week, we built an entire Linear Regression model from scratch using raw algebraic gradients. It was an amazing learning experience, but you will never do that in production.

In production, we use **Scikit-Learn** (`sklearn`). 

Scikit-Learn wraps all of that horrible calculus into two simple commands: `.fit()` and `.predict()`.

Let's look at `day6_ex2.py` and have sklearn do all the math we did last week in just three lines of code.

```python
# day6_ex2.py
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate random, noisy data
np.random.seed(42)
x = np.random.rand(100, 1) * 10
# Based on the formula: y = 3x + noise
y = 3 * x + np.random.randn(100, 1) * 2

# 1. Initialize the Scikit-Learn Model
model = LinearRegression()

# 2. Fit the Model (Scikit-Learn automatically calculates Gradients under the hood!)
model.fit(x, y)

# 3. Inspect the magically found weights
slope = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(x, y)

print("Slope: ", slope)         # Should be very close to 3!
print("Intercept: ", intercept) # Our y-intercept
print("R-Squared: ", r_squared) # How well the line fits the data!

# Visually plot the Regression Line against the scattered data
plt.scatter(x, y, color="blue", label="Data")
plt.plot(x, model.predict(x), color="red", label="Regression Line")
plt.legend()
plt.title("Linear Regression")
plt.show()
```

## Interpreting Regression
When you run `.score()` in classification, it gives you "Accuracy" (e.g., "95% correct").
When you run `.score()` in Regression, it gives you **R-Squared ($R^2$)**.

$R^2$ is a statistical measure that represents the proportion of the variance in the dependent variable that is mathematically explained by the independent variable. An $R^2$ of 1.0 means your line is utterly perfect. An $R^2$ of 0.0 means your model is drawing a flat, mathematically useless line.

## Wrapping Up Day 6
The power of `Scikit-Learn` is undeniable. All the math is hidden behind intuitive class methods.

Tomorrow, on **Day 7: The Final Project - Real-World Statistics**, we put everything together. We will load a real dataset, perform an EDA, run a Hypothesis Test, calculate the Correlations, and build a Regression Model!
