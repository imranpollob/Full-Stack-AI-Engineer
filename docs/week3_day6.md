# Day 6: Statistics Fundamentals for AI Engineers

Welcome to Day 6 of Math Week. It is not enough to simply train a neural network; you must prove your model actually works and didn't just get lucky. To do this, we use **Statistics**. 

Today, we cover Measures of Central Tendency (Mean, Median, Mode), Measures of Dispersion (Variance, Standard Deviation), and Hypothesis Testing.

## Central Tendency & Dispersion
You already know what an average is. But in Data Science, we need to know how "spread out" our data is. If the average exam score is 50%, did everyone get exactly 50%? Or did half the class get 0% and half get 100%?

**Variance** and **Standard Deviation** measure this spread.

We can calculate all of this instantly using NumPy!

```python
# day_ex1.py
import numpy as np

# A small dataset of exam scores
data = [10, 20, 30, 40, 50]

# Central Tendency
mean = np.mean(data)

# Dispersion
variance = np.var(data)
std_dev = np.std(data)

print("Mean: ", mean)
print("Variance: ", variance)
print("Standard Deviation: ", std_dev)
```

## Hypothesis Testing & P-Values
Imagine you train an AI model. Model A predicts housing prices with an error of $1000. Model B predicts them with an error of $900. Is Model B actually better? Or could that tiny difference just be random luck on the testing set?

We prove it using **Hypothesis Testing**.

There are two hypotheses:
*   **Null Hypothesis (H0):** There is NO difference between the models. Any difference is pure luck.
*   **Alternative Hypothesis (H1):** Model B is legitimately, statistically better.

When we run a statistical "T-Test," it spits out a number called a **P-Value**, which represents the probability that the Null Hypothesis is true. In Science and ML, we usually say that if the P-Value is `< 0.05` (meaning there is less than a 5% chance the results were luck), we "Reject" the null hypothesis, and accept that our AI is actually better!

Let's look at `day6_ex2.py` and run a T-Test on two groups using the `scipy.stats` library.

```python
# day6_ex2.py
from scipy.stats import ttest_ind

# Did Group 2 score significantly higher than Group 1?
group1 = [2.1, 2.5, 2.8, 3.0, 3.2]
group2 = [1.8, 2.0, 2.4, 2.7, 2.9]

# Perform independent t-test
t_stat, p_value = ttest_ind(group1, group2)

print("T-Statistic: ", t_stat)
print("P-Value: ", p_value)

# Interpretation!
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference!")
else:
    print("Failed to reject the null hypothesis: It might just be random luck!")
```

## Wrapping Up Day 6
You are now equipped with the statistical tools to rigorously evaluate data and machine learning models. You can calculate confidence intervals, standard deviations, and prove your findings with P-Values.

Tomorrow is the big one. **Day 7: Linear Regression from Scratch**.

We are going to combine the Linear Algebra matrices from Day 1, the Calculus gradients from Day 3, and the Statistical Mean-Squared-Error from Day 6 into a single project! See you then!
