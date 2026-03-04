# Day 1: Machine Learning Basics and Terminology

Welcome to Machine Learning! Before we write complex algorithms, we need to understand the language of AI. 

In traditional programming, you write the rules to process the data and output an answer. In Machine Learning, you provide the data and the answers, and the algorithm *figures out the rules itself*.

## The Three Pillars of Machine Learning

1.  **Supervised Learning:** You give the model data that already has the "answers" attached (Labeled Data). The model learns to map the data to the correct answer. This is the most common form of ML (e.g., predicting house prices based on historical sales).
2.  **Unsupervised Learning:** You just give the model a massive pile of data with no answers (Unlabeled Data). The model tries to find hidden patterns or structures itself (e.g., grouping customers into distinct segments based on buying habits).
3.  **Reinforcement Learning:** You create a digital "Agent" and put it in an environment. It tries random actions. If it does something good, it gets a mathematical reward. If it does something bad, it gets a penalty. (This is how AI learned to beat grandmasters at Chess and Go!)

## Core Terminology
When doing Supervised Learning, we use specific terms:
*   **Features (X):** The inputs. The data the model uses to make a decision (e.g., Square footage, number of bedrooms, zipcode).
*   **Target (y):** The output. The thing the model is trying to predict (e.g., The house price).

In `day1_ex1.py`, we load the "Tips" dataset. We define our `Features` (the total bill amount and party size) and our `Target` (the tip amount we want to predict).

```python
# day1_ex1.py
import pandas as pd
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
df = pd.read_csv(url)

# Define features (X) and target (y)
features = df[['total_bill', 'size']]
target = df['tip']
```

## The Golden Rule: Train/Test Splits
If you train your AI model on 1,000 houses, and then test its accuracy by asking it to predict the price of those *exact same 1,000 houses*, it will score 100%. But it didn't learn real estate; it just memorized the answers!

This is called **Overfitting**. 

To prevent this, we must ALWAYS split our data. We usually give 80% to the model to learn from (**Training Set**), and we hide the other 20% in a vault (**Testing Set**). We only test the model on the hidden 20% to see how it performs on data it has *never seen before*.

Scikit-Learn makes this incredibly easy perfectly randomly splitting the data with one function:

```python
# Split Data into 80% Training and 20% Testing!
X_train, X_test, y_train, y_test = train_test_split(
    features, 
    target, 
    test_size=0.2, 
    random_state=42 # Ensures the random split is exactly the same every time we run the script
)

print("Training Data Set Shape: ", X_train.shape)
print("Testing Data Set Shape: ", X_test.shape)
```

## Wrapping Up Day 1
You now know the vocabulary of the AI industry. You know not to test an AI on the data it was trained on.

We are ready. Tomorrow, on **Day 2: Supervised Learning & Regression**, we will take the `train_test_split` code we just wrote, and use it to train our very first predictive algorithm!
