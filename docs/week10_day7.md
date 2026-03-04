# Day 7: CNN Image Classification Capstone

Welcome to the end of Week 10! The time has come to synthesize everything you've learned to build an Enterprise CIFAR-10 Image classifier.

We abandon the ease of Keras. We will build an incredibly powerful Augmented Pipeline directly in PyTorch!

## The PyTorch Production Pipeline
Look at `day7_ex.py`. 

### Phase 1: Real-Time Transformations
PyTorch uses `transforms.Compose()` to execute real-time Image Augmentation directly on the loading pipeline to prevent Overfitting!

```python
# Create an array of random mutations executing live during the Epoch!
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),        # Randomly Mirror!
    transforms.RandomCrop(32, padding=4),     # Ensure the AI isn't dependent on borders!
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Standardize pixels!
])

# NOTE: We DO NOT Mutate the Test Set! It must remain pure and untainted!
```

### Phase 2: Dynamic Architecture
Remember how difficult it was computing the internal dimensions of PyTorch's `fc1` (Fully Connected Dense layers)?

Let's write an automated helper-function into our class to mathematically parse the Convolutional Layer outputs using a fake "Dummy Image"!

```python
class EnhancedCNN(nn.Module):
    def __init__(self):
        # ... [Define Convolutions and Batch Norms] ...
        
        # We mathematically parse the output tensor dimensions using PyTorch logic!
        self._calculate_conv_output()
        
        # We utilize that math to automatically set PyTorch's dense dimensions!
        self.fc1 = nn.Linear(self.conv_output_size, 120)
    
    def _calculate_conv_output(self):
        # We spawn a completely fake 32x32 blank image!
        dummy_input = torch.zeros(1, 3, 32, 32)
        
        # We run it through the exact convolution pipeline without calculating gradients!
        with torch.no_grad():
            output = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(dummy_input)))))))
        
        # We save the exact physical integer size of the crushed matrix output!
        self.conv_output_size = output.numel()
```

By dynamically passing a `dummy_input` tensor of dimensions $(32 \times 32)$ through the sequence, we force PyTorch to physically simulate the dimensionality loss! It passes the crushed size dynamically into `self.fc1`, meaning you never have to do mental math again!

## Wrapping Up Week 10!
Congratulations! With dynamically sized `fc` layers, integrated `BatchNorm2d`, and robust data augmentation, you have harnessed the architecture that powers Google Lens and Instagram tracking algorithms. 

You officially command Computer Vision on 2D matrices.
But the data in the world doesn't just exist as static 2D snapshots.

Data moves. Data has context. Think of the 5th word in a sentence: it is entirely dependent on the 4th, 3rd, and 2nd word! 
To process text, languages, language translations, or sequential Time-Series stock market data, CNNs fail because they lack Memory explicitly.

Next week, we graduate to **Week 11: Recurrent Neural Networks (RNNs)**. We abandon Convolutions and introduce "Networks with Memory" to process language and time itself. See you there!
