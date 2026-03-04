# Day 6: Data Visualization with Matplotlib and Seaborn

Welcome to Day 6! You can clean, merge, and group massive datasets. But if you try to show a table with 10,000 statistical averages to a client, their eyes will glaze over.

Humans understand shapes, colors, and trends. In Data Science, visualization is just as important as the math. Today, we introduce the two graphing heavyweights of Python: **Matplotlib** and **Seaborn**.

## Matplotlib: The Foundation
Matplotlib is the grandfather of Python plotting. It allows you to draw almost anything, from a simple line to 3D topographical map.

Let's look at `day6_ex1.py` to see three of the most common plots you will ever use.

### 1. The Line Plot
Used for tracking changes over time (trends).
```python
import matplotlib.pyplot as plt

years = [2010, 2011, 2012, 2013]
sales = [100, 120, 140, 160]

plt.plot(years, sales, label="Sales Trend", color="blue", marker="o")
plt.title("Sales over Years")
plt.xlabel("Years")
plt.ylabel("Sales")
plt.legend()
plt.show() # This actually launches a window to display the graph!
```

### 2. The Bar Chart
Used to compare categorical data (like the grouped data we created yesterday!).
```python
categories = ["Electronics", "Clothing", "Groceries"]
revenue = [250, 400, 150]

plt.bar(categories, revenue, color="green")
plt.title("Revenue by Category")
plt.show()
```

### 3. The Scatter Plot
Used to visualize the relationship (correlation) between two different numerical variables. Is there a relationship between hours studied and exam scores? A scatter plot shows us instantly.
```python
hours_studied = [1, 2, 3, 4, 5]
exam_scores = [50, 55, 65, 70, 85]

plt.scatter(hours_studied, exam_scores, color="red")
plt.title("Study hours vs Exam Scores")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.show()
```

## Seaborn: Advanced Visualizations
Matplotlib is powerful but can be clunky. **Seaborn** is a library built *on top* of Matplotlib. It requires much less code and generates incredibly beautiful, modern-looking statistical plots automatically.

One of the most powerful things in Seaborn is the **Correlation Heatmap**. 

If you have a dataset with dozens of columns, how do you know which columns affect each other? A Heatmap calculates the math and uses color to show you the relationships. Dark red means two variables are heavily correlated; blue means they aren't!

```python
# day6_ex2.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# We drop the String 'species' column because you can only correlate numbers!
del df['species']

# Pandas calculates the math automatically
correlation_matrix = df.corr()

# Seaborn draws the beautiful graph!
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

## Wrapping Up Day 6
You are now capable of telling a story with your data. You can show trends, plot correlations, and visualize massive datasets in a single glance.

Tomorrow is **Day 7: The EDA Project**. We will combine everything we've learned this week—from NumPy math to Pandas grouping to Seaborn plotting—into a comprehensive Exploratory Data Analysis project on a real-world dataset!
