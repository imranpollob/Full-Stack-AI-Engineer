# Day 7: Exploratory Data Analysis (EDA) Project

Welcome to the grand finale of Week 2! Today, we combine everything you've learned. NumPy for numbers, Pandas for tables, and Matplotlib/Seaborn for charts. 

When you get a brand-new dataset, the very first thing you do before writing a single line of Machine Learning code is **Exploratory Data Analysis (EDA)**. 

EDA is the process of asking questions about your data: Are there missing values? Are there typos? Which features actually matter? What are the statistical trends? 

## The Process of EDA
Every Data Scientist follows a similar workflow during EDA:
1.  **Data Cleaning**: Fix formatting errors and handle missing NaNs.
2.  **Data Transformation**: Convert variables (e.g., Dates) or generate brand new engineered features.
3.  **Aggregation & Filtering**: Group categories and calculate the Mean/Max/Sum statistics for high-level insight.
4.  **Visual Insights**: Plot the distributions, trends, and correlations to spot anomalies and generate hypotheses!

## Hands-On: The Titanic Survival Project
Let's look at `day7_ex.py`. We are going to perform a complete EDA pipeline on one of the most famous machine learning datasets in history: the passengers of the Titanic.

```python
# day7_ex.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD THE DATA
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. INSPECT THE DATA
# See how many columns we have and spot missing values
print(df.info()) 
# Instantly see the average age and fare
print(df.describe())

# 3. HANDLE MISSING VALUES (Cleaning)
# We fill missing ages with the statistical Median
df["Age"] = df["Age"].fillna(df["Age"].median())
# We fill missing Embarkation ports with the most common port (the Mode)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# 4. REMOVE DUPLICATES (Cleaning)
df = df.drop_duplicates()

# 5. FILTERING
# Let's slice the dataframe to only look at 1st Class!
first_class = df[df["Pclass"] == 1]
print("First Class Passengers: \n", first_class.head())
```

## Visualizing the Findings
Now that our data is perfectly clean, we can generate plots to answer our burning questions. 

**Hypothesis 1:** Did your passenger class affect your survival rate?
```python
# Group by Passenger Class (Pclass) and calculate the Average Survival
survival_by_class = df.groupby("Pclass")["Survived"].mean()

# We can plot a Bar chart directly from a Pandas DataFrame!
survival_by_class.plot(kind="bar", color="skyblue")
plt.title("Survival Rate by class")
plt.ylabel("Survival Rate")
plt.show()
```

**Hypothesis 2:** Was the Titanic full of young people or old people?
```python
# Let's use Seaborn to draw a beautiful Age distribution Histogram
sns.histplot(df["Age"], kde=True, bins=20, color="purple")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
```

**Hypothesis 3:** Did older people buy more expensive tickets? 
```python
# We use Matplotlib to plot a Scatter chart of Age vs Fare Price
plt.scatter(df["Age"], df["Fare"], alpha=0.5, color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
```

## Wrapping Up Week 2!
In just a week, you've gone from basic Python arrays to a full Data Science Exploratory Pipeline on a real-world dataset. You know how to wrangle messy data, compute summary statistics, and tell visual stories.

**Next Week:** We strip away the code and focus purely on the engine driving Machine Learning. In **Week 3**, we tackle the mathematics, diving deep into Linear Algebra, Calculus, and Matrix Decomposition. See you next week!
