# Day 3: Loss Functions and Backpropagation

Welcome to Day 3. Yesterday, our Neural Network pushed data Forward to make a prediction. 

But when a Neural Network is first born, all of its mathematical Weights are completely randomized. Its first prediction will always be $100\%$ wrong. 

How does it learn? We use **Loss Functions** to calculate the Error, and **Backpropagation** to distribute the blame.

## 1. The Loss Function
We must calculate exactly how wrong the AI was using a Loss metric.
*   **Mean Squared Error (MSE):** Used for Regression. If it predicted a house costs $\$100k$ but it costs $\$200k$, the difference is squared.
*   **Cross-Entropy Loss:** Used for Classification. It calculates the mathematical distance between predicted Probabilities ($90\%$ Dog) and absolute reality ($100\%$ Dog).

```python
# day3_ex.py
import numpy as np

# Binary Cross-Entropy (BCE) Loss Formula
# Logarithmically punishes the AI when it is confidently wrong!
def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15) # Prevent log(0) fatal errors
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
```

## 2. Backpropagation (The Algorithm that changed the world)
Okay, the Loss Function says our prediction was off by `.422`. 

Our network has 1 million interconnected Weights. **Which specific Weight is responsible for the error?** Did the Edge-Detector in Layer 1 cause the failure? Or the Shape-Detector in Layer 50?

We use a technique called **Backpropagation**. 
Backpropagation uses the Calculus *Chain Rule*. Starting from the Final Output Layer, it calculates the Partial Derivative (Gradient) of the Loss Function with respect to every single parameter, stepping backward through the network layer by layer. 

```python
# Derivative of MSE loss (The Calculus Gradient!)
def mse_gradient(y_true, y_pred):
    # This mathematical gradient gives us the exact DIRECTION 
    # we need to adjust the weights to lower the error!
    return 2 * (y_pred - y_true) / len(y_true)
```

It literally assigns "Blame Scores" (Gradients) to every single Neuron in the network. "Neuron A, you caused 5% of the error." "Neuron B, you caused 90% of the error."

## Wrapping Up Day 3
Backpropagation acts as the ultimate auditor. It travels backward through the network, handing every single Neuron a scorecard telling it exactly how badly it failed.

But the auditor doesn't actually fix the numbers. It just calculates the Gradients. 

Tomorrow, on **Day 4: Gradient Descent**, we introduce the Optimizers (`SGD` and `Adam`) whose sole job is to take those scorecards, look at the Gradients, and surgically alter the Weights to fix the algorithm!
