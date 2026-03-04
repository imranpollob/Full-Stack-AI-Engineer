# Day 5: Model Evaluation and Cross-Validation

Welcome to Day 5. As we discussed yesterday, "Accuracy" is a terrible metric when dealing with imbalanced data. If you are predicting fraud (0.1% of transactions) and your model has 99% accuracy, you might be totally failing to catch any fraud at all!

To truly understand how a classification model is performing, you must look at the **Confusion Matrix**.

## The Confusion Matrix
A confusion matrix is a 2x2 grid that breaks down your predictions into four categories:
*   **True Positives (TP):** Model predicted Fraud, and it WAS Fraud. (Good!)
*   **True Negatives (TN):** Model predicted Normal, and it WAS Normal. (Good!)
*   **False Positives (FP):** Model predicted Fraud, but it was Normal. (Type I Error / False Alarm)
*   **False Negatives (FN):** Model predicted Normal, but it WAS Fraud. (Type II Error / Disaster!)

From this grid, we calculate three vital metrics using Scikit-Learn (`classification_report`):
1.  **Precision:** Out of all the times the model *yelled* "Fraud!", how many times was it actually right? (Useful when False Alarms are costly).
2.  **Recall (Sensitivity):** Out of all the *actual* Fraud in the dataset, how much did the model manage to find? (Useful when missing a positive is deadly, like cancer screenings).
3.  **F1-Score:** The harmonic mean (balance) between Precision and Recall.

Let's look at `day5_ex2.py` to see a beautiful visual representation of this matrix on the famous Iris Dataset.

```python
# day5_ex2.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load Data, Train/Test Split, and Train Model... 
# (Code hidden for brevity)
y_pred = model.predict(X_test)

# Generate and Display the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.show()

# Print the full statistical breakdown!
print("Classification Report:\n", classification_report(y_test, y_pred))
```

## Cross-Validation (K-Fold)
Even if your F1-Score is high, how do you know you didn't just get lucky with an "easy" random `train_test_split`?

The gold standard for proving model stability is **K-Fold Cross-Validation**. 

Instead of splitting the data once, we split the data into $K$ chunks (usually 5). We train the model 5 separate times. Each time, we use a *different* chunk as the testing set, and the remaining 4 chunks as the training data.

If the model scores 95% on all 5 chunks, the model is incredibly robust. If it scores 95% on three chunks, but 50% on the other two... well, you have a massive variance problem.

```python
# day5_ex1.py
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(random_state=42)

# Set up the 5-Fold split mechanics
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Ask Scikit-Learn to automatically run the 5 training loops and tests!
cv_scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")

print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
```

## Wrapping Up Day 5
You now have the tools to scientifically audit a machine learning model. You can prove its accuracy across multiple folds of data, and diagnose its precision/recall weaknesses.

Tomorrow, on **Day 6: k-Nearest Neighbors**, we step away from Algebraic Regression and learn an entirely different way for AI to make decisions: Spatial Distance!
