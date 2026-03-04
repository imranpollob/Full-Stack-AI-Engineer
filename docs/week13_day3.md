# Day 3: Fine-Tuning and Data Augmentation

Welcome to Day 3. So far, we have treated our Pre-Trained backbone as an immovable, frozen rock. That is perfectly fine for basic transfer learning.

But to achieve absolute State-of-the-Art accuracy, we must perform **Fine-Tuning**.
Fine-tuning involves slowly unfreezing the top fraction of the pre-trained model and gently training it alongside our custom head. 

## The Philosophy of Unfreezing
Pre-trained CNNs capture features hierarchically:
*   **Early Layers:** Capture universal, low-level geometry (straight lines, curves, borders, colors). You should almost *never* unfreeze early layers.
*   **Middle Layers:** Capture mid-level shapes (circles, squares, basic patterns).
*   **Late Layers:** Capture highly specific, high-level features (dog snouts, airplane wings, car tires). 

If we are training a model to recognize specific types of Flowers, we want to *unfreeze* the final Late Layers of the backbone so they can "forget" what an airplane wing looks like, and re-wire themselves specifically to flower petals! 

## Hands-On: Unfreezing in PyTorch
Look at `day3_pt.py`. We download a massive pre-trained MobileNetV2. Let's look at how we selectively unfreeze layers using `named_parameters()`.

```python
# day2_pt.py (Reviewing yesterday's logic extended to unfreezing)
import torchvision.models as models

model = models.resnet50(pretrained=True)

# 1. Freeze EVERYTHING first!
for param in model.parameters():
    param.requires_grad = False

# ... [Train the custom Head for a few epochs so it stabilizes] ...

# 2. SELECTIVELY UNFREEZE Layer 4!
for name, param in model.named_parameters():
    if "layer4" in name:
        param.requires_grad = True # Reactivate Backpropagation on these specific layers!
        
# 3. CRITICAL: Continue training using an incredibly tiny Learning Rate!
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
```

Look at the `Adam` optimizer. We drop the learning rate to `1e-5`. When you unfreeze pre-trained layers, you **must use a microscopic learning rate**. If you use a normal learning rate, the erratic gradients will aggressively smash through the pre-trained weights, completely destroying the knowledge it took Google months to calculate! This is called *Catastrophic Forgetting*.

### Data Augmentation
Look at the `transforms.Compose` structure in `day3_pt.py`. To prevent our newly unfrozen layers from instantly overfitting to our tiny training set, we deploy **Data Augmentation**: 
`transforms.RandomRotation(20)`
`transforms.RandomHorizontalFlip()`

By dynamically warping the images on the fly, the model is forced to learn robust, generalized features, allowing the Fine-Tuning phase to settle perfectly.

## Wrapping Up Day 3
The ultimate Computer Vision pipeline is now complete:
1. Freeze the Backbone. 
2. Train a custom Head. 
3. Unfreeze the top layers. 
4. Drop the learning rate and gently Fine-Tune with Data Augmentation!

Now that you've mastered Vision, we pivot back to Text. Tomorrow, on **Day 4: Transfer Learning in NLP**, we apply these identical philosophies to fine-tune massive Transformer architectures!
