# Day 5: Types of Hypothesis Tests

Welcome to Day 5! We know how to use P-Values to test a hypothesis. However, the exact mathematical test you run must change depending on the *shape* of the data you are looking at. 

Are you comparing two groups of numbers? Three groups? What if you are comparing Categories (like "Male/Female" or "Red/Green/Blue") instead of numbers?

Today we cover the three major tests: **T-Tests**, **Chi-Square**, and **ANOVA**.

## 1. The T-Test Family (Comparing 1 or 2 Means)
T-Tests are used when you want to compare numerical averages (Means). There are three major flavors:

1.  **One-Sample:** Comparing a sample group to a globally known average (We did this yesterday!).
2.  **Two-Sample (Independent):** Comparing two totally different groups (e.g., Comparing the test scores of Class A vs. Class B).
3.  **Paired:** Comparing the *same* group at two different times (e.g., Weighing a group of patients before a diet, and then weighing those exact same patients after the diet).

Let's look at `day5_ex1.py` from the `Week5` folder to see all three in SciPy:

```python
# day5_ex1.py
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

data = [12, 14, 15, 16, 17]
pop_mean = 15

# 1. One-Sample
t_stat, p_value = ttest_1samp(data, pop_mean)
print("One-Sample T-Test P-Value:", p_value)

# 2. Two-Sample (Independent)
group1 = [12, 14, 15, 16, 17]
group2 = [11, 13, 14, 15, 16]
t_stat, p_value = ttest_ind(group1, group2)
print("Two-Sample T-Test P-Value: ", p_value)

# 3. Paired Relational T-Test
pre_test = [12, 14, 15, 16, 17]
post_test = [13, 14, 16, 17, 18]
# SciPy calculates the difference between each pair!
t_stat, p_value = ttest_rel(pre_test, post_test) 
print("Paired T-Test P-Value:", p_value)
```

## 2. Chi-Square Test (Comparing Categories)
What if we want to know if "Gender" strongly influences "Product Preference" (Apples vs Oranges)? You cannot calculate the "Mean" of the word 'Apple'.

Instead, we count the frequencies of each category and use the **Chi-Square ($\chi^2$) Test**. It tests if two categorical variables are mathematically independent or if they are heavily correlated!

```python
# day5_ex2.py
from scipy.stats import chi2_contingency

# A "Contingency Table" counting the frequencies
# Row 1 (e.g., Men): 50 like Apples, 30 like Oranges, 20 like Bananas
# Row 2 (e.g., Women): 30 like Apples, 40 like Oranges, 30 like Bananas
data = [[50, 30, 20], 
        [30, 40, 30]]

# Perform Chi_Square Test
chi2, p, dof, expected = chi2_contingency(data)

# If the P-Value is < 0.05, it means Gender and Fruit Preference are NOT independent!
print("P-Values:", p) 
```

## 3. ANOVA (Comparing 3+ Numerics)
What if you want to compare the test scores of Class A, Class B, *and* Class C simultaneously? You can't use a T-Test (that only handles two).

You use **ANOVA (Analysis of Variance)**. ANOVA tests the Null Hypothesis that *all* group means are completely equal. If the P-Value is low, it tells you that *at least one* group is statistically different from the others!

```python
# day5_ex3.py
from scipy.stats import f_oneway

group1 = [10, 12, 14, 16, 18]
group2 = [9, 11, 13, 15, 17]
group3 = [8, 10, 12, 14, 16]

# Perform ANOVA test
f_stat, p_value = f_oneway(group1, group2, group3)

# P-Value = ~0.67. 
# We fail to reject the null hypothesis. The groups are statistically identical!
print("P-Value:", p_value)
```

## Wrapping Up Day 5
You now know exactly which statistical test to pull out of your toolbox based on the data you are looking at:
*   2 Numeric Groups = `T-Test`
*   3+ Numeric Groups = `ANOVA`
*   Categorical Data = `Chi-Square`

Tomorrow, on **Day 6: Correlation and Regression Analysis**, we turn our attention strictly to numeric features to prepare for our final ML modeling project!
