# Day 5: Building Neural Networks in TensorFlow

Welcome to Day 5. The theoretical mathematics is behind us. Today, we harness the power of **TensorFlow** (Google) to build a sophisticated architecture natively. 

TensorFlow handles all the Forward Propagation, Backpropagation, and Adam Optimization completely transparently behind the scenes.

## The Keras API
Inside TensorFlow is an API called **Keras** (`tf.keras`). It allows you to build a Neural Network simply by stacking layers like Lego blocks!

Let's look at `day5_ex.py`. Our goal is to classify handwritten digits out of the `MNIST` dataset using a `Sequential` architecture.

```python
# day5_ex.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# 1. Initialize the Keras Architecture
model = Sequential([
    # Input Layer (Flattens 28x28 pixel images into 784 raw neurons)
    Flatten(input_shape=(28, 28, 1)),
    
    # Hidden Layer (128 Neurons using ReLU Non-Linearity)
    Dense(128, activation="relu"),
    
    # Regularization Layer (Mathematically "kills" 50% of 
    # the active neurons randomly to prevent Overfitting!)
    Dropout(0.5),
    
    # Output Layer (10 Neurons using Softmax percent conversion!)
    Dense(10, activation="softmax")   
])

# 2. Check the blueprint
model.summary()

# 3. Assemble the mathematical logic (The "Compile" phase)
model.compile(
    optimizer="adam",                  # The Math Fixer
    loss="categorical_crossentropy",   # The Error Metric
    metrics=['accuracy']
)

# 4. TRAIN THE SUPERCONDUCTOR!
history = model.fit(
    X_train, y_train,
    epochs=10,        # Rerun Forward/Backward Prop 10 times over the whole dataset!
    batch_size=32,    # Only process 32 images at a time (SGD rule)
    validation_split=0.2
)

# Output: Epoch 10/10 - val_accuracy: 0.9850 (98.5% Accuracy!)
```

## Wrapping Up Day 5
With a shocking 20 lines of code, you have built a model identifying thousands of handwritten numbers with $98\%$ accuracy! 

Because Keras handled the Calculus backpropagation, your main job is simply engineering the Architecture. You experiment by changing `128` Neurons to `256` Neurons. You experiment by modifying `Dropout(0.5)`. 

TensorFlow natively dominates enterprise AI. But there is a massive competitor built by Facebook that has completely monopolized the Research sector. 

Tomorrow, on **Day 6: PyTorch**, we will rebuild the exact same Neural Network natively inside PyTorch!
