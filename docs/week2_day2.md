# Day 2: Advanced NumPy Operations - Broadcasting and Filtering

Welcome back to Day 2! Today is where NumPy really shows off its numerical muscle. What if you try to add two matrices together that are *different* sizes? Normally, the code would crash. But NumPy has a trick up its sleeve: **Broadcasting**.

We will also learn how to instantly filter thousands of rows of data using a technique called "Boolean Indexing," and how to generate random data sets for testing purposes.

## The Magic of Broadcasting
Broadcasting is a set of rules that NumPy uses to allow operations between arrays of different shapes. It implicitly "stretches" the smaller array to match the larger one before executing the math, *without actually copying any data* (saving massive amounts of RAM!).

In `day2_ex1.py`, we demonstrate this. Notice how we add a 1D vector to a 2D matrix:

```python
# day2_ex1.py
import numpy as np

# A 3x3 Matrix
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# A 1x3 Vector
vector = np.array([1, 0 , -1])

# Broadcasting stretches 'vector' across all 3 rows of 'matrix' automatically!
result_add = matrix + vector

# The 1x1 scalar '2' stretches across the entire 3x3 matrix instantly!
result_mul = matrix * 2

print("Add: \n", result_add)
print("Multiplication: \n", result_mul)
```

## Boolean Indexing: Lightning-Fast Filtering
Let's say you have an array of 10,000 ages and you want to find everyone older than 25. If you loop through it using standard Python, it will take milliseconds. In NumPy, it takes microseconds using a boolean mask.

We can see this in action in `day2_ex2.py`. We generate a randomized 5x5 dataset containing integers between 1 and 50.

```python
# day2_ex2.py
import numpy as np

# np.random generates random numbers fast.
dataset = np.random.randint(1, 51, size=(5,5))
print("Original Dataset: \n", dataset)
```

Now, instead of a slow `for... if` loop, what if we wanted to replace any number greater than 25 with a 0? We generate a boolean condition (`dataset > 25`) and use it directly as an index!

```python
# Boolean Indexing is incredibly fast
dataset[dataset > 25] = 0
print("Modified Dataset: \n", dataset)
```

## Aggregation Functions
When working with data, you frequently need to calculate summary statistics for the entire array (or specific rows and columns). NumPy excels at this.

```python
# calculate summary stats instantly across the entire 5x5 matrix
print("Sum: ", np.sum(dataset))
print("Mean: ", np.mean(dataset))
print("Standard Deviation: ", np.std(dataset))
```

## Hands-On Let's Code!

Try taking `day2_ex2.py` one step further. Can you modify the boolean index to only find numbers that are exactly equal to 5, or numbers that are between 10 and 20? Try chaining conditions using `&` (and) and `|` (or) operators!

```python
# Example: Find all values greater than 10 AND less than 20
between_10_and_20 = dataset[(dataset > 10) & (dataset < 20)]
```

## Wrapping Up Day 2
You've now seen the true power of NumPy: executing math across multiple dimensions instantly, and filtering out thousands of data points with a single line of code.

These operations form the bedrock of AI. However, NumPy arrays only handle pure numbers. What if your data has actual text columns like "Names" and "Addresses"?

Tomorrow, on **Day 3: Introduction to Pandas**, we introduce your best friend for tabular data—the Pandas DataFrame! See you then.
