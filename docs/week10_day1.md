# Day 1: Introduction to Convolutional Neural Networks

Welcome to Week 10. Last week, we used `Dense` Networks to classify images. 

If we feed a `Dense` network a picture of a dog perfectly centered in the frame, it learns to look for "Dog Pixels" in the middle of the photograph. 

If we show that exact same AI a picture of the *same dog*, but the dog is standing in the top-left corner of the frame... the `Dense` network will fail completely. It doesn't understand that the object moved. It just sees that the middle pixels are empty.

This is a lack of **Translation Invariance**.

## The Convolutional Neural Network (CNN)
The CNN solves Translation Invariance. Instead of assigning a unique Weight to every single pixel permanently, a CNN uses a tiny sliding window (a "Kernel") that scans the entire image left-to-right, top-to-bottom. 

Because the window scans everywhere, it doesn't matter if the dog is in the center, the top-left, or the bottom-right. The window will eventually slide over it and detect the mathematical "texture" of the dog!

### The CNN Pipeline
Almost all Modern Computer Vision models follow a strict 3-Part pipeline:
1.  **Convolutional Layers:** These slide over the image and extract raw localized features (Edges, Curves, Colors).
2.  **Pooling Layers:** These downsize the image to save RAM and focus only on the brightest, most important features.
3.  **Fully Connected (Dense) Layers:** After the image is heavily processed by Convolutions, it is finally passed to a traditional Dense layer which acts as the final "Voting Classifier".

## Hands-On Let's Initialize!
Look at `day1_ex.py`. We prepare placeholders for both architectures!

```python
# day1_ex.py
import tensorflow as tf
import torch.nn as nn

# 1. TensorFlow CNN Blueprint
tf_model = tf.keras.Sequential([
    # The Sliding Window!
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    # The Downsizing Layer!
    tf.keras.layers.MaxPooling2D((2, 2)),
    # ... [Flatten and Dense Classifiers]
])

# 2. PyTorch CNN Blueprint
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # The Sliding Window!
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, activation='relu')
        # The Downsizing Layer!
        self.pool = nn.MaxPool2d(2, 2)
```

## Wrapping Up Day 1
CNNs require drastically fewer Parameters than Dense networks because a sliding Kernel (which only contains 9 weights) is re-used thousands of times across the entire image! It is beautifully efficient.

But how does a set of 9 random numbers actually extract an "Edge" from a photograph? 

Tomorrow on **Day 2: Convolutional Layers and Filters**, we drop down into the raw Numpy mathematics to manually trace an Edge-Detection matrix!
