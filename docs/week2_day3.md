# Day 3: Introduction to Pandas for Data Manipulation

Welcome to Day 3! NumPy is fantastic for pure math, but real-world data isn't just a giant block of numbers. You have customer names (Strings), timestamps (Dates), and purchases (Floats) all bundled together in spreadsheets and CSVs.

In the AI ecosystem, we organize tabular data using **Pandas**. 

## Pandas Data Structures
Pandas has two primary data structures:
1.  **Series:** A 1D column of data. You can think of it as a single column in an Excel sheet.
2.  **DataFrame:** A 2D table of data. This is simply a collection of Series side-by-side.

### Creating DataFrames
Let's manually build a tiny dataset and load it into a Pandas DataFrame using a standard Python dictionary:

```python
# day3_samples.py
import pandas as pd

# Define our data using a Dictionary
data = {"Name": ["Alice", "Bob"], "Age": [25, 30]}

# Convert it into a Pandas DataFrame
df = pd.DataFrame(data)

# Print the resulting table!
print(df)
# Output:
#     Name  Age
# 0  Alice   25
# 1    Bob   30
```

Notice the `0` and `1` on the left? That is the **Index**. Pandas automatically assigns a row number to every entry, which makes looking up data lightning fast.

## Loading and Exploring Data
You rarely create DataFrames by hand. Usually, you import massive datasets from the web or your hard drive.

Pandas does this with a single line of code. We can pull data directly from a URL or a local `.csv` file.

```python
# df = pd.read_csv("data.csv")
# df.to_csv("data.csv", index=False)
# df.to_excel("data.xlsx", index=False)
```

Once the data is loaded, how do we look at it? If you print a DataFrame with 10,000 rows, your terminal will freeze. Instead, we use exploratory methods:

*   `.head(n)`: Shows the first *n* rows.
*   `.tail(n)`: Shows the last *n* rows.
*   `.info()`: Gives you a technical summary of the dataset (columns, non-null counts, data types).
*   `.describe()`: Calculates instant statistical summaries (mean, min, max, quartiles) for all numerical columns!

## Hands-On Let's Code!

Let's look at `day3_ex1.py` to see Pandas in action. We are going to load the famous "Iris" dataset directly from GitHub.

### Exercise 1: Exploring and Filtering the Iris Dataset
The Iris dataset contains descriptions of different types of flowers. Let's load it and slice it up!

```python
# day3_ex1.py
import pandas as pd

# Load Dataset from the internet!
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# Instantly grab mathematical summaries of the flower sizes
print(df.describe())

# We can select specific columns by passing a list of column names
selected_columns = df[["species", "sepal_length"]]
print("Selected Columns: \n", selected_columns.head())
```

### Filtering Data with Conditionals
Just like we did with NumPy boolean masking yesterday, Pandas allows us to instantly filter thousands of rows. 

```python
# Let's find every 'setosa' flower with a sepal_length greater than 5.0
filtered_rows = df[(df["sepal_length"] > 5.0) & (df["species"] == "setosa")]

print("Filtered Rows: \n", filtered_rows)
```

Finally, if you need to access specific rows by their numerical index, you use `.iloc` (Index Location):
```python
# Prints the very first row
print(df.iloc[0]) 

# Prints all rows (:), but only the first column
print(df.iloc[:, 0]) 
```

## Wrapping Up Day 3
You now know how to pull data from a `.csv`, look at its summary statistics, select columns, and filter rows based on multiple conditions!

But what happens when that data is wrong? What if the spreadsheet is missing "Ages" or has formatting errors?

Tomorrow, on **Day 4: Data Cleaning**, we roll up our sleeves and learn how to handle missing values, transform columns, and merge multiple broken datasets back together!
