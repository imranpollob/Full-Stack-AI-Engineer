# Day 1: Introduction to NumPy for Numerical Computing

Welcome to Day 1 of Data Science Essentials! Today, we introduce the most important library in numerical computing: **NumPy** (Numerical Python).

Under the hood of almost every AI library (TensorFlow, PyTorch, Scikit-Learn), data is represented as arrays and matrices. Standard Python lists are great, but they are incredibly slow for complex math. NumPy solves this by executing its operations in heavily optimized C code.

## Arrays: The Foundation of NumPy
The core of NumPy is the `ndarray` (n-dimensional array). It is like a Python list, but much faster and more memory-efficient.

You can create an array from a standard list:

```python
# day1_numpybasics.py
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# Accessing elements is identical to regular Python lists
print(arr[2])  # Output: 30
print(arr[-1]) # Output: 60

# Slicing works the same way too
print(arr[1:4]) # Output: [20 30 40]
```

## Reshaping Data
In AI, we constantly change the shape of our data. For instance, you might have an image consisting of 784 pixels in a flat line, but you need to reshape it into a 28x28 square matrix. That's what `.reshape()` is for:

```python
# Turns our 1D array of 6 items into a 2D matrix (2 rows, 3 columns)
reshaped = arr.reshape(2, 3)
print(reshaped)
# Output:
# [[10 20 30]
#  [40 50 60]]
```

## Hands-On Let's Code!

Let's do some actual math using NumPy arrays.

### Exercise 1: Element-Wise Operations
If you have two Python lists and you use the `+` operator, Python *joins* the lists together. But with NumPy arrays, it performs **element-wise vector math**, which is exactly what we want!

```python
# day1_ex1.py
import numpy as np

# np.arange() operates exactly like Python's built-in range()
a = np.arange(1, 6)   # [1, 2, 3, 4, 5]
b = np.arange(6, 11)  # [6, 7, 8, 9, 10]

print("Add: ", a + b)   # Output: [ 7  9 11 13 15]
print("Sub: ", a - b)   # Output: [-5 -5 -5 -5 -5]
print("Mult: ", a * b)  # Output: [ 6 14 24 36 50]
print("Div: ", a / b)
```

### Exercise 2: Matrix Manipulations
Let's level up to 2D matrices. We can easily perform powerful math across multiple dimensions. We can also "transpose" a matrix, flipping its rows and columns!

```python
# day1_ex2.py
import numpy as np

# A 3x3 Matrix
matrix = np.array([[1, 2, 3], 
                   [4, 5, 6], 
                   [7, 8, 9]])

# Transpose the Matrix (Rows become Columns)
transpose = matrix.T
print("Transpose:\n", transpose)

another_matrix = np.array([[9,8,7], [6,5,4], [3,2,1]])

print("Addition: \n", matrix + another_matrix)
print("Multiplication : \n", matrix * another_matrix)
```

## Wrapping Up Day 1
Congratulations, you are officially doing matrix algebra in Python! As you transition into Deep Learning, these matrices (or "Tensors" as they are called in TensorFlow/PyTorch) are exactly what you will be passing into Neural Networks.

Tomorrow, in **Day 2: Advanced NumPy**, we will learn how to handle arrays of different sizes using a magical concept called "Broadcasting", and how to filter massive datasets instantly using Boolean Indexing!
