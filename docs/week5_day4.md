# Day 4: Introduction to Classification and Logistic Regression

Welcome to Day 4. Today we shift from predicting continuous numbers (Regression) to predicting discrete categories (**Classification**).

If I give an AI a photo and ask "Is this a Hotdog?", the answer is binary: Yes (1) or No (0).

## Logistic Regression 
The most famous beginner classification algorithm is ironically named **Logistic Regression**.

If you use standard Linear Regression to predict a `1` or a `0`, the mathematical line will shoot off into infinity (It might predict `45,000` or `-30`). This makes no sense for probabilities.

To fix this, Logistic Regression takes the standard Linear sum, and crushes it through a magical mathematical filter called the **Sigmoid Function**.

### The Sigmoid Function
The Sigmoid function, denoted as $\sigma(z)$, mathematically takes *any* infinite number and squashes it into a perfect range between `0` and `1`.

We can easily visualize this mathematical curve:

```python
# day4_sigmoid.py
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    # The magical formula that maps infinity to 0 - 1
    return 1 / (1 + np.exp(-z)) 

z = np.linspace(-10, 10, 100)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function")
plt.show() 
```

Because the output is always between 0 and 1, we treat it as a **Probability**. 
*   If the model outputs `0.85`, it is 85% confident the photo is a hotdog.
*   By default, the **Decision Boundary** is set to `0.5`. Anything > 0.5 is classified as a `1`, anything less is a `0`.

## Hands-On Let's Classify!
Let's look at `day4_ex1.py`. We generate synthetic data where `Age` and `Salary` determine if an individual will make a `Purchase` (1) or not (0).

```python
# day4_ex1.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ... (Synthetic Data Generation Hidden) ...

# 1. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df[['Age', 'Salary']], 
    df['Purchase'], 
    test_size=0.2, 
    random_state=42
)

# 2. Train the Logistic Regression classifier
model = LogisticRegression() # Scikit-Learn handles the Calculus!
model.fit(X_train, y_train)

# 3. Predict the classifications (Outputs arrays of 1s and 0s)
y_pred = model.predict(X_test)
```

### Classification Metrics
How do we know if it worked? We can't use MSE or $R^2$ anymore because we aren't predicting a continuous curve! We use **Accuracy**: The percentage of guesses the model got correct.

```python
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, y_pred))
# Output: Accuracy: 0.95! (95% of its predictions were correct)
```

## Wrapping Up Day 4
You have officially trained a Classification model. You can now predict binary labels and visualize the statistical decision boundaries separating your data.

But is `Accuracy` always the best metric? If you are predicting a rare disease that only affects 1% of the population, an AI that blindly predicts "No Disease" for every single person will mathematically achieve 99% accuracy—while being completely useless.

Tomorrow, on **Day 5: Model Evaluation**, we will learn about the **Confusion Matrix**, Precision, and Recall to truly audit our AI models.
