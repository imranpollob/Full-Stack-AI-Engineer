# Day 6: Regularization and Data Augmentation

Welcome to Day 6. Deep Learning models require massive amounts of data to learn correctly. 60,000 images seems like a lot, but a deep CNN will usually Overfit on that data very fast.

If an AI only memorizes exactly what those specific $60,000$ photographs look like, it will fail when passed a novel photograph.

However, gathering another $500,000$ labelled photographs is terribly expensive and time-consuming. We have to be clever.

### We artificially invent them!

## Data Augmentation
Look at `day6_ex.py`. 
We utilize TensorFlow's `ImageDataGenerator`. Instead of feeding the raw dataset straight into our Convolutional Engine, we feed it into this unique pipeline first.

```python
# day6_ex.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Build an Augmentation Engine!
datagen = ImageDataGenerator(
    rotation_range=15,       # Randomly rotate the image up to 15 degrees left or right!
    width_shift_range=0.1,   # Randomly shift the image 10% on the X axis!
    height_shift_range=0.1,  # Randomly shift the image 10% on the Y axis!
    horizontal_flip=True     # Randomly flip the image backwards like a mirror!
)

# 2. Pre-calculate the statistical limits of our specific training set!
datagen.fit(x_train)
```

## The Infinite Batch
The beauty of Data Augmentation is that it happens dynamically in real-time, purely in RAM. It doesn't permanently save 1,000,000 new JPG images to your hard drive. 

Every single time your dataset loops inside an `Epoch`, the images are dynamically scrambled!
*   **Epoch 1:** The AI sees the dog normally.
*   **Epoch 2:** The AI sees the dog slightly rotated left.
*   **Epoch 3:** The AI sees the dog shifted completely to the upper-right corner!
*   **Epoch 4:** The AI sees the dog horizontally flipped!

Because of this randomization, your network trains on a uniquely different dataset every single time. It completely destroys the mathematical ability to Overfit! The dog could be literally anywhere, upside down or backward, and the AI is forced to mathematically discover those Convolutional Edges!

## Batch Normalization
To further aid our model, we insert a brand new Layer natively into the structure: `BatchNormalization()`.

As images traverse deep through 3, 5, or 10 Convolutional blocks, the internal pixel numbers shift drastically. Batch Normalization acts like a `StandardScaler()` dynamically executed between every single Neural Network operation! It centers the matrix numbers perfectly around $0$, radically speeding up training convergence and serving as a secondary regularization fail-safe!

## Wrapping Up Day 6
The pipeline is now complete. 
*   **Convolutions** scan the entire image.
*   **MaxPooling** cuts out the noise.
*   **Batch Normalization** stabilizes the matrices.
*   **Data Augmentation** supplies near-infinite random data.

Tomorrow on **Day 7: The CNN Image Capstone**, we load CIFAR-10 natively into PyTorch and use an advanced `EnhancedCNN` architecture to synthesize the ultimate image classifier!
