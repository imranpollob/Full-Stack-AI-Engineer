# Day 7: Statistical Analysis Project - Analyzing Real-World Data

Welcome to the end of Week 4. Over the last month, we have learned the syntax of Python, the tools of Data Science, and the engine of Mathematics.

Today is our final capstone before we officially enter the world of Machine Learning algorithms next week. 

We are going to take the famous "Tips" dataset (which records waitstaff tips based on the total bill, gender of the customer, and whether they smoke), and perform a complete **Statistical Analysis Pipeline**.

## The Analytical Pipeline

### 1. Exploratory Data Analysis (EDA)
Before we write any math, we must look at the data shape. We load the CSV, ask Pandas for a summary, and plot the distribution of our primary variable (the Total Bill) using Seaborn.

```python
# day7_project.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# 2. Inspect Data
print(df.info())
print(df.describe())

# 3. Visualize Distributions
sns.histplot(df["total_bill"], kde=True)
plt.title("Distribution of Total Bill")
plt.show() # Output reveals a positive skew! Most bills are $10-$20.
```

### 2. Hypothesis Testing
Let's ask a question: *Do Men tip more than Women?* 

We cannot answer this by just looking at the average tips of men vs women in this specific restaurant. We must calculate the P-Value to see if the difference is statistically significant.

```python
from scipy.stats import ttest_ind

# Separate data by gender
male_tips = df[df['sex'] == 'Male']['tip']
female_tips = df[df['sex'] == 'Female']['tip']

# Perform Independent T-Test
t_stat, p_value = ttest_ind(male_tips, female_tips)
print("P-Value: ", p_value)

# Interpret results
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis: Significant difference.")
else:
    print("Fail to Reject the null hypothesis: NO Significant difference.")

# Output: P-Value: 0.166. 
# There is NO statistical difference in how men and women tip!
```

### 3. Correlation and Regression
Okay, if Gender doesn't predict tip size, what does? We do a quick correlation heatmap (dropping Categorical strings first) and see that `total_bill` is highly correlated with `tip`. 

Let's build a Scikit-Learn Regression model to predict exactly how much money a waiter will make based strictly on the bill amount!

```python
from sklearn.linear_model import LinearRegression

# Pandas features must be reshaped into 2D Matrices to work in sklearn!
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Output coefficients
print("Slope (Tip per Dollar): ", model.coef_[0])
print("Intercept (Base Tip): ", model.intercept_)
print("R-Squared:", model.score(X, y))

# Output:
# Slope: 0.105 (For every $1 the bill increases, the tip goes up ~10 cents!)
# R-Squared: ~0.45 
# (45% of the variance in the tip is explained purely by the bill size!)

# Plot regression
sns.scatterplot(x=df['total_bill'], y=df['tip'], color="blue")
plt.plot(df['total_bill'], model.predict(X), color="red", label="Regression Line")
plt.title("Total Bill vs Tip")
plt.legend()
plt.show()
```

## Wrapping Up Week 4!
Congratulations! You have completed the foundation. You know exactly how to handle messy data, mathematically prove your hypotheses, and predict continuous variables. 

**Next Week**: The fun truly begins. We enter **Week 5: Machine Learning Fundamentals**, where we build our very first Classification models to predict distinct categories, train Decision Trees, and learn how to properly slice data into Training and Testing sets! See you there.
