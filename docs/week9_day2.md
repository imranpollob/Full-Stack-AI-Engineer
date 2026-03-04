# Day 2: Forward Propagation and Activation Functions

Welcome to Day 2. Data only flows one way when making a prediction: Forward. From the Input Layer $\rightarrow$ Hidden Layers $\rightarrow$ Output Layer.

This flow is called **Forward Propagation**.

## The Math of a Neuron
Inside every single Artificial Neuron is a very simple linear equation you already know:
$$z = W \cdot X + b$$
*(Which is just $y = mx + b$!)*

*   **$X$:** The raw input data (e.g., the pixel brightness).
*   **$W$ (Weights):** The algorithm multiplies the pixel by a Weight. This determines how *important* that specific pixel is to this specific Neuron.
*   **$b$ (Bias):** A constant number added at the end to shift the result left or right.

## The Activation Function (The Magic)
If Neural Networks only used $y=mx+b$, the entire network would collapse into a giant, flat straight line. It could never learn a curve.

To bend the line, every Neuron passes its final result through a mathematical filter called an **Activation Function**. This adds *Non-Linearity*!

### The Big Four:
1.  **Sigmoid:** Squashes any infinite number perfectly between `0` and `1`. (Great for Output Layers predicting a flat True/False Probability).
2.  **Tanh:** Squashes any number between `-1` and `1`. 
3.  **ReLU (Rectified Linear Unit):** The absolute King of Hidden Layers. If the number is negative, it converts it to `0`. If the number is positive, it leaves it alone! This simple trick prevents math errors in deep networks.
4.  **Softmax:** Used *only* on the final Output Layer for Multi-Class problems. It takes 10 different raw numbers and perfectly converts them into 10 percentages that equal exactly 100%!

Look at `day2_ex.py`. We wrote these functions purely in NumPy!

```python
# day2_ex.py
import numpy as np

def relu(z):
    # If Z is negative, return 0. Else, return Z!
    return np.maximum(0, z)

def forward_pass(X, weights, biases, activation_function):
    # Step 1: The Linear Math (Calculate Z)
    z = np.dot(weights, X) + biases
    
    # Step 2: The Non-Linear Magic (Calculate Activation)
    a = activation_function(z)
    return a

# ... [Matrix Math Execution] ...
```

## Wrapping Up Day 2
You now know exactly how a Neural Network thinks. It multiplies inputs by weights, adds a bias, and runs it through a ReLU curve. It does this millions of times in a fraction of a second until a prediction pops out the end.

But how does it get the correct prediction? How does it know what `Weights` to use? 

Tomorrow on **Day 3: Backpropagation**, we learn how the AI uses Calculus to travel *backward* through time to fix its own math!
