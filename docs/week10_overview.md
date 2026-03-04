# Week 10: Convolutional Neural Networks (CNNs)

Welcome to Week 10 of the Full-Stack AI Engineer bootcamp! Last week, we breached the frontier of Deep Learning. We built our first Neural Networks and successfully identified rudimentary images like handwritten digits and low-resolution objects.

But our basic `Dense` architecture had a fatal flaw. When it looked at an image, it completely destroyed the geometry by flattening the 2D picture into a single 1D line of pixels.

This week, we introduce the architecture that conquered Computer Vision: the **Convolutional Neural Network (CNN)**.

## What We'll Cover This Week

*   **Day 1: Introduction to CNNs.** Why are Standard Networks bad at images? We introduce the concept of "Translation Invariance" and why an algorithm needs to recognize a dog whether it is on the left side of the photo or the right.
*   **Day 2: Convolutions & Filters.** We look at the actual mathematics of a Convolution. We learn how small $3 \times 3$ matrices called "Kernels" literally slide across an image to detect vertical edges, horizontal lines, and mathematical textures.
*   **Day 3: Pooling Layers.** Convolutions generate massive amounts of data. We learn how `MaxPooling` systematically shrinks an image without losing its most critical features, drastically speeding up the AI's processing power.
*   **Day 4: CNNs in TensorFlow/Keras.** We build our first fully-functional `Conv2D` architecture natively using the Keras API, chaining Convolutions and Pooling layers together perfectly.
*   **Day 5: CNNs in PyTorch.** We rewrite our architecture using PyTorch's `nn.Conv2d` and train it on full-color CIFAR-10 data using object-oriented classes.
*   **Day 6: Data Augmentation.** What if we only have 100 pictures of cats? We learn how to mathematically rotate, flip, and zoom those images during training to artificially generate an infinite dataset!
*   **Day 7: The CNN Capstone Project.** We bring it all together. Using PyTorch, we build a deep Convolutional network combining `BatchNorm`, `Dropout`, and `RandomHorizontalFlip` augmentations to achieve massive accuracy on CIFAR-10!

## Why This Matters
Everything from Apple's FaceID to Tesla's autonomous driving cameras relies fundamentally on Convolutional architectures. 

A Convolution doesn't just look at a pixel; it looks at the *pixels around it*. This simple addition of localized spatial awareness revolutionized the entire tech industry in 2012.

Let's teach algorithms how to see. See you tomorrow for **Day 1: Introduction to CNNs**!
