# Day 5: Probability Theory and Distributions

Welcome to Day 5. We have left the world of certainties (like $2+2=4$) and entered the world of likelihoods. 

You have to remember: AI models *don't actually know anything*. An image classifier doesn't "know" it's looking at a dog; it calculates that there is an 85% *probability* it is looking at a dog based on the statistical distribution of its training data.

Today, we learn the math of uncertainty.

## Conditional Probability & Bayes' Theorem
Conditional probability asks a simple question: "What is the probability of event A happening, *given* that event B has already happened?"

This leads directly to **Bayes' Theorem**. It allows us to update our beliefs as new evidence comes in. It is so powerful that an entire branch of Machine Learning (Bayesian Statistics) is named after it!

Let's look at `day5_ex1.py`. This is a classic example: The Medical Test.
*   Only 1% of the population has a disease. (`Prior`)
*   The test is 95% accurate if you have it. (`Sensitivity`)
*   The test is 90% accurate if you DON'T have it. (`Specificity`)

If you test positive, what is the *actual* probability you have the disease?

```python
# day5_ex1.py
def bayes_theorem(prior, sensitivity, specificity):
    # Total probability of testing positive (the 'Evidence')
    evidence = (sensitivity * prior) + ((1 - specificity) * (1 - prior))
    
    # Calculate the Posterior (The actual probability)
    posterior = (sensitivity * prior) / evidence
    return posterior

prior = 0.01 
sensitivity = 0.95 
specificity = 0.90 

posterior = bayes_theorem(prior, sensitivity, specificity)
print("Probability of Disease Given Positive Test: ", posterior)
# Output: ~0.087 (Only an 8.7% chance you actually have it!)
```
*Why so low? Because the disease is so rare, and the 10% false-positive rate on the 99% healthy population creates massive noise! AI models use these exact calculations to avoid making disastrous mistakes.*

## Probability Distributions
A probability distribution is a mathematical function that provides the probabilities of occurrence of different possible outcomes.

Different types of ML models assume different underlying distributions of data.
*   **Gaussian (Normal) Distribution:** The famous "Bell Curve." Used constantly in almost every area of ML, including Generative AI like Stable Diffusion.
*   **Bernoulli Distribution:** Models a single binary outcome (e.g., Coin flip: Heads or Tails). Used in basic Classification problems.
*   **Binomial Distribution:** Models *multiple* binary outcomes (e.g., Flipping a coin 10 times).
*   **Poisson Distribution:** Models the number of events happening in a fixed amount of time (e.g., How many customers enter a store per hour).

We can visualize all of these effortlessly using the `scipy.stats` library alongside Matplotlib!

```python
# day5_ex2.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson 

# Let's plot a Poisson Distribution
lam = 3 # Average rate of 3 events per interval
x = np.arange(0, 10)
y = poisson.pmf(x, lam)

plt.bar(x, y, label="Poisson")
plt.title("Poisson Distribution")
plt.show() # You will see the probability peak at 3, and taper off towards 10!
```

## Wrapping Up Day 5
Probability is the lens through which AI views the world. Next time your email provider automatically flags an email as "Spam," recognize that it just calculated Bayes' Theorem behind the scenes.

Tomorrow, on **Day 6: Statistics Fundamentals**, we will learn how to mathematically prove that our data (and our models) are significant and not just randomized luck!
