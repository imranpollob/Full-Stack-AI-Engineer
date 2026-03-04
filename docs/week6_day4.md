# Day 4: Feature Selection Techniques

Welcome to Day 4. Yesterday we One-Hot encoded our data. If you One-Hot encode a US State column, your dataset suddenly gains 50 new columns. 

If you have 10,000 columns (High Dimensionality), your model will train incredibly slowly and it will likely **Overfit**. 

To survive, you must become a mathematical butcher. You must ruthlessly delete useless features using **Feature Selection**.

## The Filter Methods
Filter methods evaluate features using pure statistics *before* you even start training an AI. The most common filter is the **Correlation Matrix**. Which features are mathematically correlated to your target? Keep those. Which features are strongly correlated *with each other*? Delete one of them, they are redundant!

We can also use **Mutual Information**, which measures how much "information" a feature tells us about the target. (Unlike Correlation, Mutual Information can spot non-linear relationships!). 

Let's look at `day4_ex.py` on the Diabetes dataset:

```python
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop(columns=['target'])
y = df['target']

# Calculate how much mathematical "mutual information" exists between X and y
mutual_info = mutual_info_regression(X, y)

# Create a DataFrame to view the scores
mi_df = pd.DataFrame({'Feature': X.columns, "Mutual Information": mutual_info})
mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)
print(mi_df)
# Output reveals that the 'bmi' and 's5' columns contain massive information, 
# while 'sex' and 'age' contain almost zero! We can safely delete them.
```

## The Embedded Methods (Tree Feature Importance)
The best way to know if a feature is important is to actually let the AI train, and ask the AI itself what it thought!

Decision Trees and Random Forests (`RandomForestRegressor`) natively calculate **Feature Importance** during the training phase. Every time the Tree makes a "branch" using a feature, it tracks how much that split improved the MSE. 

```python
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# 1. Train the complex Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# 2. Ask the model which features it actually used to win!
feature_importance = model.feature_importances_

importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# The AI confirms what Mutual Information told us: 'bmi' and 's5' are the absolute
# kings of this dataset, constituting 70% of the predictive power!
print(importance_df)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance from Random Forest")
plt.show()
```

## Wrapping Up Day 4
You now have the power to slash a 10,000 column dataset down to the 50 columns that actually matter. Your models will train in seconds instead of hours, and your accuracy will likely *increase* because the noise has been silenced.

Tomorrow on **Day 5: Creating and Transforming Features**, we look at what happens when the data you want is hiding inside a bad format.
