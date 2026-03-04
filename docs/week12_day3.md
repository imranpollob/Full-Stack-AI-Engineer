# Day 3: Self-Attention and Multi-Head Attention

Welcome to Day 3. So far, we've used "Attention" as a singular concept. A Query maps to a Key and spits out a Value.

But sentences are multidimensional.
`"The incredibly wealthy bank... "`

If you run a single Attention block on the word `"bank"`, what does it focus on? Does it focus on the grammar? (Bank is a noun). Does it focus on the context? (Bank relates to money). Does it focus on the emotion? 

A Single-Head Attention block will mush all of these concepts together in a blurry average.

## Multi-Head Attention
Instead of using 1 massive Attention block, we use **Multi-Head Attention**.
We mathematically split the dimensions into $8$ (or $12$, or $96$) completely standalone Attention blocks running perfectly parallel!

*   **Head 1:** Strictly learns grammatical structure (Nouns, Verbs).
*   **Head 2:** Strictly learns geographical relationships.
*   **Head 3:** Strictly learns emotional tone.
*   **Head 4:** Learns pronouns and who they map to.

Because they operate independently, the algorithm learns an unbelievably robust, nuanced understanding of language without destroying distinct contextual threads! 

## Hands-On: Building Parallel Heads in PyTorch!
Look closely at `day3_ex.py`. We initialize the PyTorch logic to physically split the dimension calculations into independent pieces!

```python
# day3_ex.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        
        # We mathematically split the Embed Dim into independent "Heads"!
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0 # Must be perfectly divisible!
        
        self.query = nn.Linear(embed_dim, embed_dim)
        # ... [Keys and Values setup]
```

### The Forward Split
Inside the `forward()` pass, the magic happens. We use `.view()` and `.transpose()` to dynamically chop the $64$-dimension vector into eight strict $8$-dimension chunks, completely isolating their calculations!

```python
    def forward(self, x):
        batch_size = x.size(0)
        
        # The Math Split! We force PyTorch to partition the Matrix!
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # We calculate the Attention Dot Products!
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        
        # We stitch the 8 Independent Heads back together into a single Output Vector!
        context = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out(context), attention_weights
```

## Wrapping Up Day 3
Multi-Head Attention is the absolute architectural bedrock of `GPT-4`, `Claude`, and `Gemini`. By deploying $96+$ independent Attention heads, models can seamlessly track sarcasm, syntax, math, and code all in the same breath.

But we have officially sidestepped a glaring problem. If all of these words are processed at the exact same time... how does the Transformer know what *order* the words were in? 

Tomorrow, on **Day 4: Positional Encoding**, we inject Trigonometry into our algorithm to solve the sequence paradox.
