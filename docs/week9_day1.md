# Day 1: Introduction to Deep Learning and Neural Networks

Welcome to Week 9. Up until today, we have used `Scikit-Learn`. Scikit-Learn is amazing for classical Machine Learning, but it cannot build Neural Networks.

Today, we graduate to the heavy artillery: **TensorFlow** (built by Google) and **PyTorch** (built by Meta/Facebook).

## What is a Neural Network?
An Artificial Neural Network (ANN) is a massive sequence of mathematical equations loosely inspired by the biological brain. 

### The Architecture
1.  **Input Layer:** The raw data. If you have an image that is 28x28 pixels, your Input Layer has $784$ unique input "Neurons" (one for every single pixel).
2.  **Hidden Layers:** The magic happens here. The data is passed through thousands of interconnected equations. The first layer might detect "edges." The second layer combines edges into "shapes." The third layer combines shapes into "faces."
3.  **Output Layer:** The final guess! If we are classifying digits from $0$ to $9$, the output layer has exactly $10$ Neurons predicting the probability of each number!

## The Datasets of Deep Learning
Classical ML uses datasets like the "Titanic" CSV. Deep Learning requires millions of data points, so we use standardized benchmark datasets:

*   **MNIST:** 60,000 images of handwritten numbers (0-9). The "Hello World" of Deep Learning.
*   **CIFAR-10:** 60,000 full-color images of 10 different objects (Airplanes, Dogs, Frogs).

Let's look at `day1_ex.py`. We load both TensorFlow and PyTorch, download the datasets, and define our very first Artificial Neurons!

```python
# day1_ex.py
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import torch.nn as nn

# 1. Download the Massive "Hello World" image dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f"MNIST Dataset: Train - {X_train_mnist.shape}") 
# Output: (60000, 28, 28) -> 60,000 images that are 28x28 pixels!

# 2. Define a layer of 10 Artificial Neurons using Google's TensorFlow
tf_layer = tf.keras.layers.Dense(units=10, activation='relu')

# 3. Define a layer of 5 Artificial Neurons using Meta's PyTorch
torch_layer = nn.Linear(in_features=10, out_features=5)
```

## Wrapping Up Day 1
A Neural Network is just a massive matrix of numbers. But how do those numbers actually combine a pixel into a prediction? 

Tomorrow on **Day 2: Forward Propagation**, we are going to dive into the mathematical formulas powering the network. We will learn how a `Weight`, a `Bias`, and an `Activation Function` turn data into intelligence.
