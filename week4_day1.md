# Day 1: Probability Theory and Random Variables

Welcome to Day 1 of Probability and Statistics Week. We begin with the foundation of uncertainty: **Probability**. 

When an AI tries to translate a sentence from English to French, it doesn't "know" the right word. It calculates the *probability* of every possible word and picks the one with the highest score.

## Basic Probability Concepts
*   **Sample Space:** The set of all possible outcomes. (e.g., A six-sided die has a sample space of `{1, 2, 3, 4, 5, 6}`).
*   **Event:** A specific outcome or subset of outcomes we care about. (e.g., Rolling an even number: `{2, 4, 6}`).
*   **Independence:** Two events are independent if the outcome of one does not affect the outcome of the other. If you flip a coin twice, the first flip has zero effect on the second flip!

We can simulate probability effortlessly with `numpy`:

```python
# day1_ex1.py
import numpy as np

# Simulate rolling a 6-sided die 10,000 times!
rolls = np.random.randint(1, 7, size=10000)

# What is the probability of rolling an Even number?
# (Sum up all the True values, and divide by the total number of rolls)
P_even = np.sum(rolls % 2 == 0) / len(rolls)
print("P(Even): ", P_even) 
# Output: ~0.50 (As expected, 50% chance!)

# What is the probability of rolling greater than a 4? (5 or 6)
P_greater_than_4 = np.sum(rolls > 4) / len(rolls)
print("P(Greater than 4): ", P_greater_than_4)
# Output: ~0.33 (33% chance!)
```

## Random Variables
A **Random Variable (RV)** is a mathematical way to map the outcome of a random event to a number. 
*   **Discrete RVs** have specific, countable values. (e.g., The number rolled on a die).
*   **Continuous RVs** can be *any* value within a range. (e.g., The exact height of a human).

RVs are defined by their curves:
*   A **PMF (Probability Mass Function)** shows the probability curve for Discrete variables.
*   A **PDF (Probability Density Function)** shows the curve for Continuous variables.

```python
# day1_ex2.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

# We can plot the PDF of a Continuous "Uniform" curve
# A uniform curve means every single number has the exact same probability of being picked!
x = np.linspace(0, 1, 100)
pdf = uniform.pdf(x, loc=0, scale=1)

plt.plot(x, pdf, color="red")
plt.title("PDF of Uniform(0,1)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.show() # Draws a perfectly flat red line!
```

## Expectation and Variance
If you play a game of chance infinitely, what is your average mathematical outcome? This is called your **Expectation** (or Mean).

But how "wild" is the game? Does it always result in a number close to the mean, or does it swing wildly between massive wins and massive losses? That spread is the **Variance** (and its square root, the **Standard Deviation**).

```python
# day1_sample.py
import numpy as np

# A fair 6-sided die
outcomes = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6]* 6)

# The Expectation is the Sum of (Outcome * Probability of that Outcome)
expectation = np.sum(outcomes * probabilities)
print("Expectation (Mean): ", expectation) 
# Output: 3.5 

variance = np.sum((outcomes - expectation)**2 * probabilities)
std_dev = np.sqrt(variance)

print("Variance: ", variance) # Output: ~2.91
print("Standard Deviation: ", std_dev) # Output: ~1.70
```
This tells us that the average roll is a 3.5, and we expect most rolls to deviate from that average by roughly 1.7!

## Wrapping Up Day 1
You now understand the difference between discrete and continuous variables, and how to calculate mathematically what you "expect" to happen in a scenario involving uncertainty.

But these were simple, flat distributions. Real world data isn't flat; it clusters into beautiful, predictable bell curves. Tomorrow, on **Day 2: Probability Distributions in ML**, we will dive deep into Gaussians!
