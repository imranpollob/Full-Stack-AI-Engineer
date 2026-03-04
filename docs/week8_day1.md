# Day 1: Introduction to Hyperparameter Tuning

Welcome to Week 8. Before we optimize models, we must define the difference between the two types of numbers living inside an AI.

## Parameters vs. Hyperparameters
1.  **Parameters:** These are the numbers the AI *learns itself* during training. Examples include the internal Weights and Biases of a Neural Network, or the Coefficients in a Linear Regression equation ($y = mx + b$, where $m$ and $b$ are learned). **You do not touch these.**
2.  **Hyperparameters:** These are the architectural settings that **YOU** set *before* training begins! They control the environment the AI learns inside. If you set them poorly, the AI will fail to learn.

### Common Hyperparameters
*   **Max Depth** (Random Forest): How many layers deep is the decision tree allowed to grow?
*   **Number of Estimators** (Random Forest): How many independent trees should exist in the forest?
*   **Learning Rate** (Gradient Boosting): How violently should the algorithm update its internal math when it makes a mistake? 

## Hands-On Let's Tune!
Look at `day1_ex.py`. We load the Breast Cancer dataset and train a `RandomForestClassifier`.

First, we use the default Hyperparameters chosen by Scikit-Learn:
```python
# Train Random Forest with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

print(f"Default Model Accuracy: {accuracy_score(y_test, rf_default.predict(X_test)):.4f}")
# Output Default Accuracy: ~0.9649
```
It scores $96.4\%$ accuracy. Not bad! 

But the default Random Forest allows trees to grow to **infinite depth** until every single leaf is homogenous! This causes severe overfitting. 

Let's manually intervene. Let's force the forest to build *more* trees (400 instead of 100), but restrict their depth to a maximum of 5 layers so they are forced to generalize!

```python
# Train Random Forest with adjusted hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=400, # Massive forest
    max_depth=5,      # Shallow trees (Prevents overfitting!)
    random_state=42
)
rf_tuned.fit(X_train, y_train)

print(f"Tuned Model Accuracy: {accuracy_score(y_test, rf_tuned.predict(X_test)):.4f}")
# Output Tuned Accuracy: ~0.9737
```
By simply changing two architectural numbers, our accuracy jumped to $97.3\%$! 

## Wrapping Up Day 1
We manually proved that altering Hyperparameters improves the model. But we simply guessed `400` and `5`. What if the actual optimal numbers were `328` and `7`? 

It would take us a week to manually test every number by hand. Tomorrow on **Day 2: Grid Search and Random Search**, we let algorithms do the guessing for us!
