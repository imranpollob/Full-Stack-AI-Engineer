# Day 6: k-Nearest Neighbors (k-NN) Algorithm

Welcome to Day 6. So far we have classified things using Logistic Regression— drawing a mathematical line through the data. 

But what if the data isn't cleanly divisible by a line? What if the Red Dots are completely surrounded by a ring of Blue Dots? A straight line will mathematically fail.

Enter **k-Nearest Neighbors (k-NN)**.

## How k-NN Works
k-NN does *not* draw lines. In fact, it doesn't do any math during the `fit()` stage at all! It just memorizes where every data point is physically located in space.

When you ask the model to predict a new, unknown data point, the model simply uses the Pythagorean theorem to calculate the physical Distance between the new point and every point it memorized. 

It looks at its $k$ closest neighbors. If $k=5$, it looks at the 5 closest dots. If 4 of them are "Spam" and 1 of them is "Not Spam", the model predicts "Spam" by a majority vote!

## Hands-On Let's Classify!
Let's look at `day6_ex1.py`. We load the Iris dataset and compare Logistic Regression to k-NN.

**Crucial Step:** Because k-NN calculates physical distance, you MUST scale your features! If one feature is measured in "Miles" and another is measured in "Inches", the "Inches" feature will mathematically overpower the distance calculation. We use `StandardScaler` to put everything on an even playing field.

```python
# day6_ex1.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALE THE DATA! (Extremely important for Distance-Based Algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Predict using k-NN with k=5
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)

# The model just memorizes the training data here
knn.fit(X_train, y_train)

# The model does all its heavy lifting (calculating distances) here!
y_pred_knn = knn.predict(X_test)

print(f"k-NN Accuracy (k={best_k}): ", accuracy_score(y_test, y_pred_knn))
```

## Choosing $k$
How do we know the golden number for $k$?
*   **If $k$ is too small (e.g., k=1):** The model is incredibly sensitive to noise. If there is a single random corrupted data point, the model will follow it blindly. (Overfitting).
*   **If $k$ is too large (e.g., k=1000):** The model ignores local patterns and just predicts the most common class in the entire dataset. (Underfitting).

We usually test multiple values of $k$ in a `for` loop, using Cross-Validation to see which number yields the best F1-Score! A solid starting guess is usually the square root of the number of training samples.

## Wrapping Up Day 6
You have learned an "Instance-Based" algorithm. It's incredibly powerful for non-linear data, but it has a major weakness: it is physically slow. Calculating the distance to 10 million data points takes a huge amount of computing time!

Tomorrow is the big finale: **Day 7: Supervised Learning Mini Project**. You will pit multiple models against each other in a head-to-head competition!
