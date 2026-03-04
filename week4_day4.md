# Day 4: Hypothesis Testing and P-Values

Welcome to Day 4! Suppose an AI algorithm trained to detect fraud flag an account, and the customer complains. How do you defend your algorithm's choice mathematically?

You use **Hypothesis Testing**.

## The Core Concept
Hypothesis testing is exactly like a court of law. In a trial, a defendant is presumed "Not Guilty" (the Null Hypothesis) until the prosecutor provides so much mathematical evidence that the jury is forced to convict them (accepting the Alternative Hypothesis).

*   **Null Hypothesis ($H_0$)**: "There is NO effect. It was just random chance."
*   **Alternative Hypothesis ($H_1$)**: "There IS a mathematical effect."

## P-Values and Alpha Levels
When you run a statistical test on your data, the test spits out a **P-Value**. 

The P-Value answers one incredibly specific question: *Assuming the Null Hypothesis is true, what is the mathematical probability of seeing the data we're looking at?*

If the P-Value is very small (say, `0.02` or 2%), it means: "There is only a 2% chance this data occurred by random luck." 

Do we accept a 2% chance? That threshold is called our **Alpha ($\alpha$)**. The global standard for Alpha is `0.05` (5%). If the P-Value drops below Alpha, we "Reject the Null Hypothesis" and declare a statistical victory!

## Hands-On: A One-Sample T-Test
Let's apply this. In `day4_ex1.py`, we have a class of 7 students. We know the global average test score is 15. The average of our sample is actually 15.8! 

Did our class statistically perform "better" than the global average? Or was it just luck because our sample is so tiny?

We use `ttest_1samp` from SciPy to test our sample against the known Population Mean.

```python
# day4_ex1.py
import numpy as np
from scipy.stats import ttest_1samp

# Sample data
data = [12, 14, 15, 16, 17, 18, 19]

# Null Hypothesis: True global mean is 15.
population_mean = 15

# Calculate the T-Statistic and the P-Value
t_stat, p_value = ttest_1samp(data, population_mean)
print("T-Statistic: ", t_stat)
print("P-Value: ", p_value)

# Interpret Results against our Alpha threshold (5%)
alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to Reject the null hypothesis: no significant difference")

# Output: Fail to Reject! P=0.31
# There is a massive 31% chance our students' "higher" score was just luck. 
# We cannot statistically claim they are smarter!
```

## Two-Sample T-Test
What if we don't know the global average? What if we just want to know if Group A scored better than Group B?

We use a Two-Sample (Independent) T-Test. Looking at `day4_ex2.py`:

```python
# day4_ex2.py
from scipy.stats import ttest_ind

group1 = [12, 14, 15, 16, 17, 18, 19]  # Mean: 15.8
group2 = [11, 13, 14, 15, 16, 17, 18]  # Mean: 14.8

# Run the independent test
t_stat, p_value = ttest_ind(group1, group2)

alpha = 0.05
if p_value <= alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to Reject the null hypothesis: no significant difference")

# Output: Fail to Reject! P=0.55
# With a P-Value of 55%, the difference between these two groups is entirely negligible.
```

## Type I and Type II Errors
The court of law isn't perfect. Neither is math. Because we use a 5% Alpha threshold, it means 5% of the time, our math will lie to us.

*   **Type I Error (False Positive):** We convicted an innocent man. We rejected the Null Hypothesis when it was actually true!
*   **Type II Error (False Negative):** We let a guilty man go. We failed to reject the Null Hypothesis even though there really *was* an effect.

## Wrapping Up Day 4
You now know the absolute gold standard metric of scientific research: The P-Value. You know exactly what an Alpha of 0.05 means.

But T-Tests only work when calculating numerical Averages. What if your data is composed of Categories? Or what if you have three groups instead of two?

Tomorrow on **Day 5: Types of Hypothesis Tests**, we expand our toolkit to include Chi-Square and ANOVA!
