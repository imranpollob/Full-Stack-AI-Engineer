# Day 3: Statistical Inference - Estimation and Confidence Intervals

Welcome to Day 3. Today we discuss the most powerful magic trick in statistics: **Inference**.

Imagine you want to know the average height of every human on Earth (the *Population*). It is physically impossible to measure 8 billion people. But, through the math of Statistical Inference, if you measure a *Sample* of just a few thousand people, you can predict the global average with stunning accuracy.

## Point Estimation vs. Interval Estimation
*   **Point Estimate:** A single guess. (e.g., "The average height is exactly 5'8\""). This is almost guaranteed to be wrong.
*   **Interval Estimate:** A range. (e.g., "The average height is between 5'7\" and 5'9\""). This is much safer, but how mathematically confident are we?

This leads us to the **Confidence Interval (CI)**.

## The Confidence Interval
A Confidence Interval provides a range of values within which the true population "Mean" is likely to lie. Standard practice in AI and science is to use a **95% Confidence Interval**.

This means: "If I took 100 random samples and calculated the interval each time, the true global average would be inside my interval 95 times."

### How to Calculate a CI
Let's look at `day3_ex1.py`. We have a Sample of 100 numbers. We want to know the true Mean.

The formula requires three things:
1.  **Sample Mean**: The simple average of our 100 data points.
2.  **Standard Error**: How spread out is our data? We calculate this using the Standard Deviation divided by the square root of $n$ (our sample size).
3.  **Z-Score**: A mathematical constant dictating how confident we want to be. For 95% confidence on a normal distribution, the Z-score is roughly `1.96`. We can grab the exact number using `scipy.stats.norm.ppf`.

```python
# day3_ex1.py
import numpy as np
from scipy.stats import norm

# Generate 100 random data points (This is our "Sample"!)
data = np.random.normal(loc=50, scale=10, size=100)

# Calculate the Sample Mean and Standard Deviation
mean = np.mean(data)
# ddof=1 means we use "Bessel's correction" for a sample rather than a population!
std = np.std(data, ddof=1) 
n = len(data)

# Calculate the 95% Confidence Interval
z_value = norm.ppf(0.975) # This gets the ~1.96 Z-score!
margin_of_error = z_value * (std / np.sqrt(n))

ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean: ", mean)
print("95% Confidence Interval: ", ci)
# Output: (48.1, 52.0)
# We isolated the true average using only 100 data points!
```

## The T-Distribution
What happens if you have an incredibly small dataset? Imagine you are building a medical AI and you only have trial data for 7 patients. 

If your Sample Size ($n$) is less than 30, the Z-Score math breaks. Instead, William Sealy Gosset invented the **T-Distribution** (under the pseudonym "Student"). It makes the confidence interval wider to account for the massive uncertainty of a tiny sample!

```python
# day3_sample.py
import numpy as np
from scipy.stats import t

# Sample Data (n = 7)
data = [12, 14, 15, 16, 17, 18, 19]

mean = np.mean(data)
std = np.std(data, ddof=1)
n = len(data)

# Use the Student's T distribution! df = degrees of freedom (n-1)
t_value = t.ppf(0.975, df=n-1)
margin_of_error = t_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error) 

print("95% Confidence Interval: ", ci)
```

## Wrapping Up Day 3
Whenever you report a metric in Machine Learning (like "My model achieves 92% accuracy"), you should always include a Confidence Interval ("92% accuracy ± 1.5%"). It proves you understand that your test set was just a sample!

Tomorrow on **Day 4: Hypothesis Testing**, we take everything we learned about intervals and use it to formally prove hypotheses about our data using the fabled **P-Value**!
