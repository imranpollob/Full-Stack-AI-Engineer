# Day 5: LightGBM and CatBoost

Welcome to Day 5. XGBoost is incredible, but in 2017 two major tech corporations released competing algorithms that improved upon XGBoost in highly specific use-cases.

## 1. LightGBM (Microsoft)
XGBoost grows trees "Level-Wise" (it splits the tree evenly on both sides).
**LightGBM** grows trees "Leaf-Wise". If the math detects that splitting the left branch drops the global Error drastically, it will completely ignore the right branch and hyper-focus on deepening the left branch. 

*   **Pros:** LightGBM is brutally fast. It requires significantly less RAM than XGBoost, and usually trains 2x to 5x faster on massive datasets (millions of rows). 
*   **Cons:** Because Leaf-Wise growth is so aggressive, it is incredibly prone to Overfitting on datasets with fewer than 10,000 rows. Do not use it on tiny datasets!

## 2. CatBoost (Yandex)
XGBoost only accepts numbers. If you have "Categorical" text (like `['Blue', 'Red']`), you must `OneHotEncode` them. But what if you have high-cardinality text (like $10,000$ unique Zipcodes)? If you OneHot encode it, your dataset explodes to $10,000$ columns, and XGBoost grinds to a halt.

**CatBoost** natively understands, parses, and encodes String Categories internally using advanced Target Statistics! You just hand it raw Text strings, and it mathematicians the answer perfectly. 

## Hands-On Let's Classify!
Look at `day5_ex.py` on the Titanic Dataset. 
We can literally just tell CatBoost which columns contain strings (`['Sex', 'Embarked']`), and pass the DataFrame directly in without using a `ColumnTransformer`!

```python
# day5_ex.py
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load raw Pandas data completely unencoded!
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
X = df[['Pclass','Sex','Age','Fare', 'Embarked']]
y = df['Survived']

# Instantiate the CatBoost algorithm (Tell it which Columns are Raw Strings)
cat_features = ['Sex', 'Embarked']
cat_model_native = CatBoostClassifier(cat_features=cat_features, verbose=0)

# Train directly on text!
cat_model_native.fit(X_train, y_train)

# Predict!
cat_preds_native = cat_model_native.predict(X_test)
print(f"CatBoost Native Accuracy: {accuracy_score(y_test, cat_preds_native):.4f}")
```

## When to use Which?
The rule of thumb for modern tabular data is:
1.  **High-Cardinality Strings?** (Thousands of Zipcodes, Cities, Names). Use `CatBoost`.
2.  **Massive Dataset?** (Millions of rows). Use `LightGBM`.
3.  **Everything Else?** (Standard sized, numerical/encoded generic data). Use `XGBoost`. 

## Wrapping Up Day 5
You are now armed with the "Big Three" gradient boosting algorithms of modern Data Science! We can conquer huge sets of data instantly.

But what happens when the target is mathematically lopsided? Tomorrow on **Day 6: Handling Imbalanced Data**, we learn about the mathematical problem of "99% Normal 1% Fraud", and how algorithms fail under that pressure.
