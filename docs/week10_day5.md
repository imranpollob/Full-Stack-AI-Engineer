# Day 5: Building CNN Architectures with PyTorch

Welcome to Day 5. Now that you understand the 3-part Pipeline of a CNN (Conv $\rightarrow$ Pool $\rightarrow$ Dense), we must implement it utilizing **PyTorch**.

As we learned in Week 9, PyTorch requires you to explicitly implement your own Object Oriented Forward Propagation Loop. 

But PyTorch has another massive quirk: **You must calculate the mathematics yourself.**

## PyTorch Layers
Look at `day5_ex.py`. We initialize the exact same CNN architecture we used yesterday!

```python
# day5_ex.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Part 1: Convolution taking 3 color channels, outputting 6 mathematical filters
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Part 2: Max Pooling
        self.pool = nn.MaxPool2d(2, 2)
        # Part 3: Secondary Convolution taking the 6 mathematical filters!
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Part 4: The Dense Architecture!
        # WHY IS THE INPUT 16 * 5 * 5?
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

Look intimately at `self.fc1`. This is the first Dense layer. 
When passing a flattened image into Keras, Keras dynamically calculates how big the image is after all the Convolutions and sets the Linear dimension manually.

In PyTorch, **you must mentally track the integer size of the image through every single step of the pipeline!** 
We passed in a $32 \times 32$ image. After the convolutions stripped out padding, and the `MaxPool2d` cut it in half twice, the final image size was exactly $5 \times 5$ pixels across $16$ filters. 
If we used $16 \times 6 \times 6$, the network would crash fatally during the Forward Pass!

## The Forward Pass Pipeline
In PyTorch, you physically apply the activation functions to the layers natively in the explicit math sequence!

```python
    def forward(self, x):
        # 1. Image hits Conv1 -> Returns Z.
        # 2. Z hits Relu -> Returns A.
        # 3. A hits MaxPool -> Returns compressed Z.
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # 4. View() acts identically to Flatten()!
        x = x.view(-1, 16 * 5 * 5)
        
        # 5. Pass into the Linear Dense Predictors!
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

## Wrapping Up Day 5
PyTorch trades ease of use for absolute explicit granular control. You can precisely monitor the image dimensional loss, or even inject custom tensor calculations right into the middle of the `forward()` pass.

But wait. If we train our algorithm, it still struggles to crack $80\%$ accuracy. 

Our model has seen the 60,000 CIFAR-10 images exactly $20$ times. To get better accuracy, we cannot just add more Layers. To break $80\%$, we need physically *more data*.

Tomorrow on **Day 6: Data Augmentation**, we learn how to synthetically generate millions of invisible fake photographs using transformations!
