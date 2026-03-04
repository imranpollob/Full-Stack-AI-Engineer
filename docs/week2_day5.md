# Day 5: Data Aggregation and Grouping in Pandas

Welcome to Day 5! So far, we've learned how to slice rows, filter math, clean data, and merge tables. But when doing Machine Learning, you're not trying to understand a single data point; you're trying to understand **trends**.

How do you aggregate millions of data points into summary trends? Today, we unlock one of the most powerful features of Pandas: **Grouping Data**.

## Grouping Data by Categories
If you have a dataset with categorical features (like "City" or "Student Grade") and numerical features (like "Sales" or "Test Score"), you will usually want to aggregate the numerical features *based on* the categorical ones.

Pandas does this with `.groupby()`. Think of it like a SQL `GROUP BY` statement.

When you use `.groupby()`, Pandas physically splits the DataFrame into smaller separate chunks based on your categories, applies a mathematical function to those chunks, and then glues them back together.

```python
# day5_samples.py
# Splitting the frame into groups
grouped = df.groupby("column_name")

# If you iterate over the groups, you get the chunk name and the chunk DataFrame!
for name, group in grouped:
    print(name)
    print(group)
```

## Aggregation Functions
Once data is grouped, what is the math you want to do?
*   `.sum()`: Totals. (e.g., Total Sales per City)
*   `.mean()`: Averages. (e.g., Average Student Score per Grade)
*   `.max()` / `.min()`: Highest and Lowest values.

Let's look at today's exercise (`day5_ex1.py`):

```python
# day5_ex1.py
import pandas as pd

# Creating a dataset of Students in different Classes
data = {
    "Class": ["A", "B", "A", "B", "C", "C"],
    "Score": [85, 90, 88, 72, 95, 80],
    "Age": [15, 16, 15, 17, 16, 15],
}

df = pd.DataFrame(data)
print("Original Dataset \n", df)

# 1. Group by "Class" and calculate the mathematical Mean of all remaining columns
grouped = df.groupby("Class").mean()
# print(grouped)
# Output:
#        Score   Age
# Class             
# A       86.5  15.0
# B       81.0  16.5
# C       87.5  15.5
```
As you can see, Pandas instantly aggregates every single student in Class A, Class B, and Class C and calculates their average Age and test Score!

## Multi-Aggregation (`.agg()`)
What if you want to know the `.mean()`, the `.min()`, and the `.max()` of your groups simultaneously? You pass a dictionary of functions into the incredibly powerful `.agg()` method.

```python
# Pass a dictionary where the keys are the columns
# and values are lists of math functions string names!
stats = df.groupby("Class").agg({
    "Score": ["mean", "max", "min"], 
    "Age": ["mean", "max", "min"]
})

print(stats)
```

## Pivot Tables & Custom Aggregations
If you've ever used Excel's Pivot Tables, Pandas has them too! 
```python
pivot = df.pivot_table(
    values="numeric_column",
    index="category_column",
    aggfunc="mean"
)
```

Want to run a mathematical function that isn't built into Pandas? Just write a standard Python function and pass it directly to `.agg()`!

```python
# Custom Python function to calculate the literal Range of data points
def range_func(x):
    return x.max() - x.min()

df.groupby("category_column")["numeric_column"].agg(range_func)
```

## Wrapping Up Day 5
By combining filtering from Day 3, merging from Day 4, and grouping from Day 5, you have reached the summit of Data Wrangling. 

But tables of statistical averages are difficult to read and impossible to put into a PowerPoint presentation. Tomorrow, on **Day 6: Data Visualization**, we will learn how to turn Pandas Grouping tables directly into beautiful line charts, bar plots, and heatmaps!
