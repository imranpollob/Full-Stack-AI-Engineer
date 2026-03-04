# Day 5: Cross-Validation and Model Evaluation Techniques

Welcome to Day 5. We discussed Cross Validation (CV) deeply in Week 5. The gold standard for Model Evaluation is splitting your data into $K$ blocks, training the model $K$ independent times, and averaging the results.

But what happens if your data is highly imbalanced (E.g., 99% Normal, 1% Cancer)?

## The Danger of Standard K-Fold
If we use a standard `KFold(n_splits=5)`, Scikit-Learn will blindly chop our dataset into 5 random chunks. 

Because the data is $99\%$ Normal, it is highly likely that one of those random 5 chunks will contain absolutely **zero** examples of Cancer! 
If we use that chunk as the internal Test set, the algorithm calculates an Accuracy of $100\%$ that is completely fraudulent. 

## Stratified K-Fold
We fix this using **Stratification**. 
A `StratifiedKFold` forces Scikit-Learn to statistically analyze the global dataset first. It mathematically ensures that *every single one of the 5 chunks perfectly matches the global distribution*. 

If the global distribution is $99\%$ Normal / $1\%$ Fraud, then every single chunk will be exactly $99\%$ Normal / $1\%$ Fraud!

## Hands-On Let's Stratify!
Look at `day5_ex.py` on the Credit Card Fraud dataset!

```python
# day5_ex.py
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# 1. Initialize a dangerous, blind K-Fold (BAD for Imbalanced Data)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Evaluate the Random Forest using standard K-Fold
rf_model = RandomForestClassifier(random_state=42)
scores_kfold = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy')

print(f"K-Fold Mean Accuracy: {scores_kfold.mean():.2f}")


# 3. Initialize a Stratified K-Fold (PERFECT for Imbalanced Data)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Evaluate safely!
scores_stratified = cross_val_score(rf_model, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Stratified Mean Accuracy: {scores_stratified.mean():.2f}")
```

While the Average accuracy might look mathematically identical ($0.99$), the real power of Stratification is in the Variance! 

If you look at the raw array of the standard `scores_kfold`, you might see massive unstable swings: `[0.99, 1.00, 0.96, 0.99, 1.00]`.
If you look at `scores_stratified`, it is mathematically stabilized: `[0.99, 0.99, 0.99, 0.99, 0.99]`.

## Wrapping Up Day 5
If you are doing Regression, use `KFold`.
If you are doing Classification, use `StratifiedKFold`. 

And guess what? When you call `GridSearchCV(cv=5)` on a classification target, Scikit-Learn automatically uses Stratified K-Fold under the hood! 

Tomorrow on **Day 6: Automated Tuning Algorithms**, we combine Grid Search with advanced algorithmic paradigms like Support Vector Machines!
