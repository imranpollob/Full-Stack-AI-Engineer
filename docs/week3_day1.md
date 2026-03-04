# Day 1: Linear Algebra Fundamentals

Welcome to Day 1 of Math Week! Today, we start with Linear Algebra. 

When you feed an image into an AI model, it doesn't "see" a picture of a cat. It sees a massive grid of numbers representing pixel values. Everything in Deep Learning—text, audio, video—is eventually converted into these grids, known as Tensors (or Matrices). Linear algebra is the study of how to manipulate these grids.

## Vectors and Matrices
A **Vector** is a 1-dimensional array of numbers. Think of it as a coordinate pointing somewhere in space.
Example: `[2, 3, 4]`

A **Matrix** is a 2-dimensional grid of numbers (rows and columns). It is simply a stack of vectors.
Example:
```text
[ 2, -3,  1 ]
[ 2,  0, -1 ]
[ 1,  4,  5 ]
```

## Fundamental Operations
We don't do these operations by hand; we use the tool we learned last week: NumPy. Let's look at `day1_ex1.py` and `day1_ex2.py` to see the math in action.

### Addition and Scalar Multiplication
Adding two matrices is simple: you just add the corresponding elements together. Same with scalar multiplication (multiplying a matrix by a single number).

```python
# day1_ex1.py
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[9, 8], [7, 6]])

# Adds 1+9, 2+8, etc...
print("Addition\n", A + B)

# Multiplies every number in A by 3
print("Scalar Mult: \n", 3 * A)
```

### Matrix Multiplication (The Dot Product)
Matrix multiplication isn't as simple as element-wise multiplication. It is the complex product of rows and columns (called the Dot Product). In neural networks, computing the dot product between your input data and the network's "weights" is the core calculation.

We do this using NumPy's `np.dot()` function.

```python
# day1_ex2.py
import numpy as np

# A 3x3 Matrix
M = np.array([[1, 2, 3], 
              [4, 5, 6], 
              [7, 8, 9]])

# A 1D Vector with 3 elements
v = np.array([1, 0, -1])

# Calculate the algebraic dot product!
result = np.dot(M, v)
print("Matrix-Vector Multiplication: \n", result)
# Output: [-2 -2 -2]
```
*(Notice how multiplying a 3x3 Matrix by a 3-element vector results in a new 3-element vector? This dimensionality shift is exactly how AI layers change the shape of data!)*

## Special Matrices
There are a few mathematical matrices that act as special tools.

*   **Zero Matrix:** A matrix full of exactly zeroes. Useful for initializing an empty AI model.
*   **Identity Matrix ($I$):** The matrix version of the number `1`. Any matrix multiplied by the Identity Matrix just equals itself!
*   **Diagonal Matrix:** A matrix where all non-zero values are only on the main diagonal line.

```python
# day1_ex3.py
import numpy as np

# Creates a 3x3 Identity Matrix (1s on the diagonal, 0s everywhere else)
I = np.eye(3)

# Any matrix A multiplied by I remains exactly A.
# A = np.dot(A, I)

D = np.diag([1, 7, 9]) # Creates a diagonal matrix
Z = np.zeros((3, 3))   # Creates a 3x3 Zero matrix
```

## Wrapping Up Day 1
You have successfully defined and multiplied mathematical matrices in Python. 

While NumPy hides the complexity, understanding *why* you are using `np.dot()` instead of `*` is crucial for debugging model architecture errors later on!

Tomorrow on **Day 2: Advanced Linear Algebra**, we discuss Determinants, Inverses, and Eigenvalues!
