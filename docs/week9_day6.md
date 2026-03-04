# Day 6: Building Neural Networks with PyTorch

Welcome to Day 6. Yesterday we used TensorFlow. TensorFlow abstracts everything away from the user—you literally just type `.compile()` and the AI trains itself.

Enter **PyTorch**. 
PyTorch forces you to embrace the flow. You must manually define the Layers. You must manually command the Gradients. You manually step through the Forward and Backward prop loops! 

Because it gives developers such absolute granular control over the data flow, PyTorch is used to train roughly 90% of all modern Large Language Models (`LLaMA`, `OpenAI`, etc).

## Writing a PyTorch Neural Network
Look at `day6_ex.py`. We classify MNIST handwritten digits again, but this time, PyTorch style!

### The Model Engineering
Unlike Keras' elegant `Sequential` lists, PyTorch forces you to use `Object Oriented Programming`. You must literally define a Class and write the custom `.forward()` flow logic yourself.

```python
# day6_ex.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. We must define a custom Object Inheriting torch attributes!
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define the math blocks
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)  # Hidden Layer 1
        self.fc2 = nn.Linear(128, 64)       # Hidden Layer 2
        self.fc3 = nn.Linear(64, 10)        # Output Layer
        
    def forward(self, x):
        # We manually pipe the data through the layers and activation matrices!
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
model = NeuralNetwork()
```

### The Training Loop
There is no `model.fit()` in PyTorch. You write the loops by hand!

```python
# The Rules of Engagement
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop manually over 5 Epochs!
for epoch in range(5):
    for images, labels in train_loader:
        # 1. Wipe the gradients from the previous batch!
        optimizer.zero_grad()
        
        # 2. FORWARD PASS!
        outputs = model(images)
        loss = criterion(outputs, labels) # Calculate error!
        
        # 3. BACKWARD PASS! (The Calculus kicks in natively!)
        loss.backward()
        
        # 4. GRADIENT DESCENT! (Fix the weights!)
        optimizer.step()
```

## Wrapping Up Day 6
If you want to rapidly prototype tabular or production models, use TensorFlow. 
If you want to build Custom Architecture logic, Transformers, or State-of-the-Art research algorithms, you *must* master PyTorch. 

Tomorrow on **Day 7: The Final Project**, we scale up! We use full-color images from the CIFAR-10 dataset and train a massive network to guess animals and machines accurately using Regularization strategies!
