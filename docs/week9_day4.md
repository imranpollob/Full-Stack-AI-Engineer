# Day 4: Gradient Descent and Optimization Techniques

Welcome to Day 4. So far, our Neural Network has predicted an outcome. `MeanSquaredError` measured how wrong it was. `Backpropagation` assigned Calculus Gradients to every single Weight in the network.

Now, the final step of learning begins. We must mathematically alter the Weights using the **Optimizers**.

## 1. Gradient Descent (The Basics)
Gradient Descent is the original Optimizer. It looks at the Calculus Gradient attached to a specific weight. 
If the Gradient is positive, scaling the Weight *down* lowers the global error. 
If the Gradient is negative, scaling the Weight *up* lowers the global error!

The algorithm structurally walks opposite to the Gradients to reach the mathematical "bottom" of the bowl—the absolute minimal error state.

```python
# day4_ex.py
import numpy as np
m = 100
theta = np.random.rand(2, 1) # Random Weights

# Stochastic Gradient Descent!
for iteration in range(1000): # 1000 Epochs!
    
    # 1. Backpropagate the Error using Calculus (The Gradients)
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    
    # 2. OPTIMIZE! Adjust the original Weights based on the gradients!
    theta -= learning_rate * gradients
```

### The Learning Rate
Look closely at the code above: `learning_rate = 0.1`. 
The Learning Rate is a Hyperparameter multiplying the Gradients. If the learning rate is `1.0`, the algorithm takes massive steps and completely overshoots the answer. If the learning rate is `0.0001`, the algorithm takes microscopically tiny, safe steps but might take 2 weeks to finish learning!

## 2. Advanced Optimizers (Adam)
Standard Gradient Descent is fine for simple datasets, but it struggles on completely randomized, chaotic image datasets.

Deep Learning revolves around three advanced versions of Gradient Descent:
1.  **Stochastic Gradient Descent (SGD):** Instead of calculating the gradients using the full dataset (slow), it calculates the gradients using only $32$ rows (a "Batch"). It is blazingly fast but mathematically wobbly.
2.  **RMSprop:** Instead of blindly jumping down the bowl, it remembers how fast it jumped during the *previous* steps using exponential decay!
3.  **Adam (Adaptive Moment Estimation):** The King. It combines SGD momentum with RMSprop memory! It dynamically alters the learning rate internally during training perfectly on the fly!

## Wrapping Up Day 4
You have now completed the conceptual core of Deep Learning: 

1.  **Forward Propagation** guesses an answer using Weights and ReLUs.
2.  **Backpropagation** calculates the errors using Calculus.
3.  **Gradient Descent** uses `Adam` to mathematically fix the Weights.

These three steps are run thousands of times seamlessly in a loop known as an **Epoch**.

Tomorrow, on **Day 5: TensorFlow and Keras**, we dump theoretical Numpy math and finally build a true deep neural network using Google's frameworks.
