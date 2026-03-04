# Day 2: Advanced Linear Algebra Concepts

Welcome to Day 2 of Math Week. Yesterday we talked about constructing matrices. Today, we talk about destroying them. By breaking a matrix down into its fundamental mathematical properties, we can compress data, reduce noise, and solve complex equations instantly.

We will cover Determinants, Inverses, Eigenvalues, and Matrix Decomposition.

## Determinants and Inverses
A **Determinant** is a scalar value (a single number) that can be calculated from a square matrix. If you think of a 2x2 matrix as representing a 2D shape, the determinant tells you the *scaling factor* of that shape's area. Crucially, if the determinant is `0`, the matrix is "singular" and has no inverse.

The **Inverse** of a matrix $A$ is denoted as $A^{-1}$. If you multiply a matrix by its inverse, it perfectly cancels out, leaving you with the Identity Matrix ($I$). This is how you "divide" matrices in linear algebra!

NumPy handles this effortlessly using the `np.linalg` (Linear Algebra) module.

```python
# day2_ex1.py
import numpy as np

# A 3x3 Matrix
A = np.array([[2, 3, 4], 
              [4, 5, 6], 
              [7, 8, 9]])

determinant = np.linalg.det(A)
inverse = np.linalg.inv(A)

print("Determinant: ", determinant)
print("Inverse: \n", inverse)
```

## Eigenvalues and Eigenvectors
When you multiply a matrix by a vector, the vector usually changes direction and size. However, for every square matrix, there are special vectors that *do not* change direction when multiplied by the matrix—they only stretch or shrink. 

These are **Eigenvectors**, and the amount they stretch is the **Eigenvalue**.

Why do we care? Because these vectors represent the "principal components" or the most important directions of variance in our data. This math entirely powers algorithms like Principal Component Analysis (PCA) used for dimensionality reduction!

```python
# day2_ex2.py
import numpy as np

A = np.array([[4, -2],
              [1,  1]])

# NumPy calculates both instantly
eigvals, eigvec = np.linalg.eig(A)

print("EigenValues: ",eigvals)
print("EigenVectors: \n",eigvec)
```

## Singular Value Decomposition (SVD)
What happens if your matrix isn't perfectly square? You can't calculate Eigenvectors. Instead, you use SVD.

SVD is a matrix decomposition technique that breaks *any* data matrix into three simpler matrices. It is widely used in AI for image compression and recommendation systems (like Netflix guessing what movie you want to watch).

SVD decomposes matrix $A$ into: $U \cdot \Sigma \cdot V^T$

```python
# day2_ex3.py
import numpy as np

A = np.array([[3, 1, 1], [-1, 3, 1], [1, 1, 3]])

# Calculate SVD
U, S, Vt = np.linalg.svd(A)

print("U:\n", U)
print("Singular Values:\n", S)
print("V Transpose:\n", Vt)

# We can perfectly reconstruct our original matrix!
Sigma = np.zeros((3, 3))
np.fill_diagonal(Sigma, S)

# The '@' symbol in Python is shorthand for np.dot() matrix multiplication!
reconstructed = U @ Sigma @ Vt
print("Reconstructed Matrix \n", reconstructed)
```

## Wrapping Up Day 2
Matrix decomposition and eigenvalue calculations are mathematically dense topics. The good news is that you will rarely have to calculate them by hand. The goal is to understand *what* the math is doing to your data.

Tomorrow on **Day 3: Calculus for Machine Learning**, we switch gears from algebra to the math of change. We will learn how AI models actually learn iteratively using Derivatives and Gradients!
