# Day 1: Introduction to Feature Engineering

Welcome to Week 6! You now know how to run `model.fit(X, y)`. But what exactly goes into `X`?

The columns of your dataset are called **Features**. 
**Feature Engineering** is the process of transforming raw, messy world data into robust mathematical inputs that make it as easy as possible for an algorithm to learn cleanly.

## The Three Types of Features
Before we can engineer a feature, we must identify what type of data it holds:

1.  **Numerical Features:** Numbers holding mathematical value. (e.g., Age, Salary, Temperature).
2.  **Categorical (Nominal) Features:** Text or groupings that have *no mathematical order*. (e.g., "Red", "Blue", "Green", or "Male", "Female"). You cannot say "Red is greater than Blue".
3.  **Ordinal Features:** Categorical data that *does* have a strict mathematical order! (e.g., "Low", "Medium", "High", or "Small", "Large"). We *can* say High > Low.

## Identifying Features with Pandas
When you load a new dataset, you should immediately ask Pandas to separate the numbers from the text. 

Let's look at `day1_ex.py`. We load the famous Titanic dataset, which contains information about passengers.

```python
# day1_ex.py
import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Separate features mathematically using their Pandas Data Type!
# "object" usually means Strings/Text (Categorical)
categorical_features = df.select_dtypes(include=["object"]).columns

# "int64" and "float64" mean Numbers (Numerical)
numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

print("Categorical Features: ", categorical_features.tolist())
# Output: ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

print("Numerical Features: ", numerical_features.tolist())
# Output: ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
```

Once we have separated them, we can analyze them individually. We can look at the mathematical averages of the Numerical features:
```python
print(df[numerical_features].describe())
```

And we can count the frequency of the Text features:
```python
for col in categorical_features:
    print(f"{col}:\n", df[col].value_counts(), "\n")
```

## Wrapping Up Day 1
Whenever you get a new dataset, step one is sorting the columns into `Numerical` and `Categorical` lists. 

Why? Because you have to treat them completely differently!

Tomorrow on **Day 2: Data Scaling & Normalization**, we will learn how to handle the `Numerical` list. (If one numerical column is Age (0-100) and another is Salary (0-200,000), distance-based algorithms will break!).
