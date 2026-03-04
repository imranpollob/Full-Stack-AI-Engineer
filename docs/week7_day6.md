# Day 6: Handling Imbalanced Data

Welcome to Day 6. Imagine you work for a credit card company. You have $1,000,000$ transactions. 
*   $999,000$ are Normal. 
*   $1,000$ are Fraud.

If you train a Random Forest on this dataset, the AI will learn a terrifying truth: "If I just blindly guess *Normal* every single time, I will mathematically achieve an Accuracy score of 99.9%!" 

The AI will output a phenomenal Accuracy score on the training set while being completely useless at actually detecting Fraud. This is the danger of **Imbalanced Data**. 

## Technique 1: Class Weights
To fix this, we can tell the algorithm to violently penalize mistakes made on the Fraud class. If the algorithm gets a Normal prediction wrong, it loses $1$ point. If it gets a Fraud prediction wrong, it loses $1,000$ points. 

Scikit-Learn supports this natively by adding `class_weight="balanced"` to almost every algorithm classifier! 
```python
# The model will internally multiply the importance of Fraud errors to match Normal!
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
```

## Technique 2: SMOTE
Sometimes weight penalization isn't enough, especially if you only have 5 instances of Fraud data in total. The AI just doesn't have enough physical examples to learn the patterns of a fraudster.

**SMOTE (Synthetic Minority Over-sampling Technique)** solves this structurally. It analyzes the 5 Fraud rows, and mathematically interpolates the distances between them in latent space (using k-Nearest Neighbors!) to synthesize *completely brand new, fake Fraud data*.

SMOTE literally invents synthetic Fraud transactions, injecting them into the training set until $50\%$ of the data is Fraud and $50\%$ is Normal!

Look at `day6_ex.py` on the Imbalanced Credit Card Fraud dataset:

```python
# day6_ex.py
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
# Normal counts: 284,315. Fraud counts: 492
y = df['Class']

# 2. Resample Data! Synthesize Fake Fraud Rows!
# This ONLY applies to the Training set!
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 3. Prove they are balanced!
print(pd.Series(y_resampled).value_counts())
# Output:
# Normal counts: 227451. Fraud counts: 227451 (Magic!)

# 4. Train model safely!
rf_model_smote = RandomForestClassifier()
rf_model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, rf_model_smote.predict(X_test)))
```

## Wrapping Up Day 6
With SMOTE, your models will finally look at the Minority class and dedicate massive learning resources to mapping those hidden statistical trends, massively increasing your `Recall` scores!

Tomorrow, on **Day 7: The Super Showdown**, we combine everything from Week 6 and 7 into a massive pipeline comparing all three Boosting algorithms simultaneously!
