# Day 7: The Neural Network Capstone

Welcome to the end of Week 9! You understand PyTorch, TensorFlow, Forward propagation, and Gradient Descent parameters. 

Today we abandon Black and White handwritten digits. We load the legendary **CIFAR-10 Dataset**. 
It contains 60,000 full-color images of 10 different target classifications (Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck).

## The CIFAR-10 Challenge
Look at `day7_ex.py`. 

Full-color images are massively complex. Instead of just 1 color channel (Grayscale 28x28), they have 3 color channels (Red, Green, Blue at 32x32).

If we flatten this out, our basic `Dense` network will struggle to maintain spatial relationships between pixels resulting in severe overfitting. 

### Overfitting Defeat
To counter the sheer complexity of color logic, we must introduce Regularization. 
We design an incredibly deep Architecture utilizing Keras' `Dropout` layer. 

By randomly severing $50\%$ of all mathematical connections between Hidden Layer 1 and the Output layer, we mathematically force the AI to robustly learn multiple representations of a "Dog", because it can never rely on a single dominant pixel path to give it the answer!

```python
# day7_ex.py

# ... [Load Data] ...

# 1. An ultra-dense, modernized Keras architecture mapping spatial filters!
improved_model = Sequential([
    # Input
    Conv2D(64, (5, 5), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    
    # Internal Deep Hidden Blocks
    Conv2D(128, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    
    # Fully Connected Blocks!
    Dense(256, activation='relu'),
    
    # REGULARIZATION! 50% Connection Severance!
    Dropout(0.5),
    
    # Output! 10 Distinct Classes
    Dense(10, activation='softmax')
])

# 2. Compile using a perfectly optimized Adam Step Size
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optimizer, 
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# 3. Train the monster across 20 Epochs!
history = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

# Output test Accuracy ranges from ~68% to ~75%!
```

## Wrapping Up Week 9!
Congratulations! You have successfully built a Deep Learning algorithm that can identify a frog out of a photograph!

But wait... $75\%$ accuracy? Can we do better? 

There is an advanced architectural trick you noticed scattered throughout our `Sequential` layout. It is called `Conv2D`—a **Convolution**. 

Dense Layers destroy spatial relationships by flattening images into a 1D line. Convolutions use tiny $3 \times 3$ sliding windows to literally scan 2D images for "Texture" shapes, completely eliminating spatial loss! 

Next week on **Week 10: Convolutional Neural Networks (CNNs)**, we dive specifically into the bleeding-edge architecture responsible for Computer Vision! See you there!
