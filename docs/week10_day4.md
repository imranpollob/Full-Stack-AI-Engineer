# Day 4: Building CNN Architectures with Keras and TensorFlow

Welcome to Day 4. Today we build a completely functional Convolutional Neural Network (CNN) using `TensorFlow` and `Keras`!

We return to our `CIFAR-10` dataset containing 60,000 $32 \times 32$ pixel images of 10 different objects (Airplanes, Dogs, Frogs, etc.).

## The Architecture
Let's assemble the $3$-part pipeline discussed on Day 1 using `day4_ex.py`. 

```python
# day4_ex.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Initialize the Keras Architecture
model = Sequential([
    # PART 1: The First Feature Extraction Block
    # 32 Different sliding windows (kernels) processing 3-Color Channel input!
    Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)),
    # Compress the (32x32) image down into a (16x16) array of brightest features!
    MaxPooling2D((2, 2)),
    
    # PART 2: The Deep Feature Extraction Block
    # Look deeper! Now use 64 different sliding windows on the compressed data!
    Conv2D(64, (3, 3), activation='relu'),
    # Compress the (16x16) data down into a tiny (8x8) array!
    MaxPooling2D((2, 2)),
    
    # PART 3: The Dense Voting Classifier!
    # Destroy all 2D Matrix structures and Flatten into a 1D Python list.
    Flatten(),
    # Create 128 Artificial Neurons to look at the list of features!
    Dense(128, activation='relu'),
    # Randomly kill 50% of the connections natively to prevent overfitting!
    Dropout(0.5),
    # The Output! 10 Neurons mapped specifically to percentages (Softmax)
    Dense(10, activation='softmax')
])

model.summary()
```

When you print `model.summary()`, look closely at the **Output Shape**. 
1. Our image started as $32 \times 32$.
2. The first `Conv2D` shaved off the outer edge pixels, making it $30 \times 30$.
3. The first `MaxPooling2D` cut that cleanly into $15 \times 15$.
4. The second `Conv2D` shaved more edge pixels making it $13 \times 13$.
5. The second `MaxPooling2D` cut that into a minute $6 \times 6$ image!

By the time the data reached our `Dense` layer, it was completely unrecognizable to a human being, containing only intense blocks of math highlighting distinct features of a "Frog" vs a "Dog".

## The Compiler
Because Keras is built on top of TensorFlow, compiling and training the network requires just three lines of code using the `categorical_crossentropy` Loss Function and the `Adam` mathematical optimization engine!

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the Convolution Scanner!
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)
```

## Wrapping Up Day 4
You have successfully built an industry-standard image classification model! 
Keras makes development blazingly fast by completely masking the underlying matrices from you. You literally construct Lego blocks, and TensorFlow seamlessly handles mathematical dimensionality behind the scenes.

But you don't control the math.

Tomorrow on **Day 5: CNNs in PyTorch**, we will rewrite this massive pipeline natively and take explicit absolute control over the data flow variables!
