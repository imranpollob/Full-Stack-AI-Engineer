# Day 6: Model Evaluation Techniques

Welcome to Day 6. You have built a dataset, cleaned it, transformed it, and trained a model. How do you summarize the model's accuracy to a manager? 

You must select the correct evaluation metric.

## 1. Regression Metrics (Predicting Numbers)
When you predict continuous numbers (like House Prices), your model's guesses will almost never be 100% exactly correct to the decimal. Thus, we categorize our predictions by their "Error".

*   **MAE (Mean Absolute Error):** The easiest to explain. It calculates the average difference between your prediction and reality. "On average, my AI was off by $10,000."
*   **MSE (Mean Squared Error):** Before taking the average, it mathematically squares every single error. This heavily penalizes massive outliers! If the AI guesses $100 off, the error is $10k. 
*   **RMSE (Root Mean Square Error):** Identical to MSE, but takes the square root at the very end to bring the number mathematically back down to reality.
*   **R-Squared ($R^2$):** A percentage score of how much mathematical variance your line successfully modeled. Usually between 0 and 1. 1.0 means perfect.

```python
# day6_ex2.py
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ... Train linear regression model ...
y_pred = model.predict(X_test)

# Calculate exactly how bad the errors are!
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2-Score(R2): {r2_score(y_test, y_pred):.2f}")
```

## 2. Classification Metrics (Predicting Categories)
As we learned last week, predicting "Yes/No" or "Spam/Not Spam" requires a Confusion Matrix because a flat "Accuracy" score hides False Positives and False Negatives.

*   **Precision:** Out of all the times the model shouted "Spam!", how many times was it correct? (Minimizes False Alarms).
*   **Recall:** Out of all the real Spam on the planet, how much did the model flag? (Minimizes missing terrible events, like disease).
*   **F1 Score:** The mathematical balance between Precision and Recall.

In `day6_ex.py`, we construct a Confusion Matrix on the Iris dataset:
```python
# day6_ex.py
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# ... Train LogisticRegression model ...
y_predict = model.predict(X_test)

# Create the visual grid!
cm = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Class 0", "Class 0"])
disp.plot(cmap="Blues")

# Print the complete breakdown of every True/False Positive!
print(classification_report(y_test, y_predict))
```

## Are False Positives or False Negatives Worse?
The only way to answer this is Business Logic.
If you are classifying Cancer imagery, a False Positive is annoying (they take another test), but a False Negative is lethal. You must tune your model for **Recall**, even if it hurts your Precision!

If you are classifying Spam emails, a False Negative is annoying (you delete an email), but a False Positive means your boss's critical email goes to the Spam folder unseen! You must tune your model for **Precision**, even if it hurts your Recall!

## Wrapping Up Day 6
Tomorrow is **Day 7: Cross Validation and Hyperparameter Tuning**. You will learn the ultimate Data Science trick: giving your algorithm a list of different mathematical knobs, and having it test them autonomously to find the highest score!
