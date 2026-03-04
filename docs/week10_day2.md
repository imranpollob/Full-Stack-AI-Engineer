# Day 2: Convolutional Layers and Filters

Welcome to Day 2. Today we crack open the "Sliding Window" to see the math inside. 

The window is officially called a **Kernel** or a **Filter**. It is usually a very small $3 \times 3$ grid of numbers. 

When this $3 \times 3$ grid is placed over an image, it mathematically multiplies its 9 internal numbers by the 9 pixels directly underneath it! It sums those 9 results together to produce a single new pixel.

## The Magic of Kernels
If the 9 numbers inside the Kernel are completely randomized, it will produce garbage. 
But if the 9 numbers are *specifically arranged*, they act as mathematical enchantments!

Look at `day2_ex.py`. We manually define two famous Filters purely in Numpy without using any Deep Learning libraries!

```python
# day2_ex.py
import numpy as np
from scipy.ndimage import convolve

# 1. The Edge Detection Kernel!
# Look at the numbers! It heavily multiplies the center pixel by 8,
# but it subtracts the surrounding pixels! 
# This mathematically isolates stark contrast boundaries (Edges!)
edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])

# 2. The Blur Kernel!
# This simply averages out every pixel with its immediate neighbors!
blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9 

# When you apply these kernels to an image, the raw math literally redilutes the pixels!
edge_detected_image = convolve(image, edge_detection_kernel)
```

## The Goal of Deep Learning
In the 1990s, scientists had to manually type out these Kernel arrays by hand to detect edges. 

The absolute genius of a Convolutional Neural Network is that **we do not type the numbers.**
We initialize 64 Kernels with completely blank random numbers, and **Backpropagation learns the numbers!** 

Over thousands of epochs, the AI will slowly mutate Kernel #1 into an Edge Detector, Kernel #2 into a Color isolator, and Kernel #63 into a "Dog Ear" identifier!

## Important Hyperparameters
When adding a `Conv2D` layer in TensorFlow or PyTorch, you must define:
1.  **Filters:** How many Kernels do you want to use simultaneously? (E.g., 32 or 64). Think of this as giving the AI 64 different pairs of glasses to look at the image with.
2.  **Kernel Size:** How big is the sliding window? Convolution usually uses $3 \times 3$ or $5 \times 5$.
3.  **Stride:** How many pixels does the window move every step? (Default is 1).

## Wrapping Up Day 2
By running an image through 64 different Filters, your AI generates 64 different mathematical "Feature Maps". 

This gives the AI incredible understanding, but it requires a massive amount of RAM and Processing power. Tomorrow, on **Day 3: Pooling Layers**, we introduce the algorithm used to safely compress and shrink those feature maps!
