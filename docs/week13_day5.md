# Day 5: Fine-Tuning Techniques in NLP

Welcome to Day 5. Yesterday, we used the Hugging Face `Trainer` abstraction. Today, we step out of the black box and manually engineer our own PyTorch loop.

Why? Because NLP models are incredibly delicate. 

If you apply a static $0.001$ learning rate across a 110-Million Parameter pre-trained Transformer on Epoch 1, the massive initial Gradients originating from your random Classification Head will crash backward through the model and immediately destroy Google's underlying grammatical matrices. This is called **Catastrophic Forgetting**.

## Preventing NLP Collapse
To meticulously control the Fine-Tuning mathematics, Researchers invented dynamic training algorithms.

### 1. Discriminative Fine-Tuning
Instead of applying a single learning rate across the entire model, we physically separate the layers! We apply a tiny learning rate ($1e-5$) to the deepest layers (which understand basic grammar), and a slightly larger learning rate ($1e-3$) strictly to the final Classification Head, ensuring exactly the right amount of plasticity at every level.

### 2. Slanted Triangular Learning Rates (STLR)
A static learning rate is dangerous. STLR introduces a dynamic, sweeping arc over the entire Training process!

Instead of remaining static, we physically force the learning rate to:
1.  **Warm-Up Phase:** Linearly ramp UP from $0.0$ to $2e-5$ over the first $10\%$ of training. This gently introduces new concepts to the model without exploding gradients.
2.  **Decay Phase:** Linearly ramp DOWN from $2e-5$ back down to $0.0$ over the remaining $90\%$ of training! As the model iterates, we slowly force it to converge on its final mathematical state perfectly without overshooting!

## Hands-On: PyTorch Scheduler Control
Look closely at `day5_ex.py`. We initialize the algorithm inside standard PyTorch!

```python
# day5_ex.py
from transformers import AutoModelForSequenceClassification, get_scheduler
import torch

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# 1. We mathematically calculate exactly how many Steps the model will train!
num_training_steps = len(train_loader) * 3  # 3 Epochs

# 2. We dictate precisely 10% for the initial "Warm-up" phase!
warmup_steps = int(0.1 * num_training_steps)

# 3. WE INITIALIZE THE SCHEDULER (STLR)!
scheduler = get_scheduler(
    "linear", 
    optimizer=optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=num_training_steps
)

def train_model():
    model.train()
    for batch in train_loader:
        # Standard PyTorch Boilerplate
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        optimizer.step()
        
        # 4. CRITICAL: Trigger the Scheduler! Modify the Learning rate Dynamically!
        scheduler.step()
        optimizer.zero_grad()
```

By explicitly injecting `scheduler.step()` inside the training loop, the Learning Rate physically alters itself every single millisecond, slowly accelerating, then slowly decaying, guaranteeing perfect Fine-Tuning synchronization!

## Wrapping Up Day 5
By abandoning the Hugging Face `Trainer` and utilizing pure PyTorch, we unlock absolute dynamic control over our Model's learning plasticity! 

But what if despite perfect fine-tuning, the math *still* fails? 
If you train a normal BERT model on Medical DNA Text, no amount of STLR scheduling will save it. The fundamental embedding vocabulary won't map to biology. 

Tomorrow, on **Day 6: Domain Adaptation**, we learn how to definitively hot-swap the foundational Vocabulary mapping by utilizing ultra-specific domain models like `BioBERT`.
