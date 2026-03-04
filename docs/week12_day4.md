# Day 4: Positional Encoding and Feed-Forward Networks

Welcome to Day 4. As we learned, Transformers do not read sequentially like an LSTM. Every single word in the paragraph is passed through the `MultiHeadAttention` blocks at the exact same millisecond. 
Because the algorithm is 100% parallelized, it processes data 500x faster than Recurrent architectures! 

But the paradox of parallelization: if all the words enter the matrix simultaneously, the matrix has zero concept of "before" and "after". 

"The dog bit the man" and "The man bit the dog" would produce the exact same Semantic Matrix value.

## Enter: Positional Encoding
To fix this, researchers introduced a brilliant hack. We physically alter the Semantic Embedding Coordinates before they ever enter the Attention mechanism. 

We mathematically inject **Sine** and **Cosine** trigonometric waves directly into the Embedding!

1.  We assign a unique mathematical Phase Shift to the first word ($Word \ 1 = \operatorname{Sin}(0)$).
2.  We assign a mathematically predictable shift to the second word ($Word \ 2 = \operatorname{Sin}(0.14)$).
3.  We inject predictable offset frequencies across all 1,000 words.

Because the waves perfectly oscillate and weave through the Dimensions, the Neural Network quickly learns that `[Coordinate X + Sin(0.14)]` definitively comes *before* `[Coordinate Y + Sin(0.28)]` !

## Hands-On: The Trigonometric Injection
Look at `day4_ex.py`. We initialize the sine/cosine mathematics directly via PyTorch Tensors!

```python
# day4_ex.py
import numpy as np

# 1. The Mathematical Wave Generator
def positional_encoding(seq_len, embed_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(embed_dim)[np.newaxis, :]
    
    # Calculate the frequency decay to stretch the waves over long sequences!
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates
    
    pos_encoding = np.zeros(angle_rads.shape)
    
    # Apply SINE to even indices!
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # Apply COSINE to odd indices!
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return pos_encoding
```

### Passing the Waves into the Tensor Chassis
Now, we must physically inject the math into the Transformer's Input sequence. We use standard Tensor addition inside the PyTorch `.forward()` pass!

```python
class TransformerWithPositionalEncoding(nn.Module):
    # ... [init setup]
    
    def forward(self, x):
        # 1. We load the Stanford GloVe Embeddings.
        # 2. WE INJECT THE SINE WAVES DIRECTLY INTO THE EMBEDDINGS!
        x = self.embedding(x) + self.positional_encoding
        
        # 3. NOW we pass the mathematically altered coordinates into the Attention Engine!
        attn_output, _ = self.multihead_attention(x, x, x)
        
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x
```

## Wrapping Up Day 4
Positional Encoding is the elegant workaround that gave algorithms perfect "Memory" without forcing them to run slowly sequentially. 

By running $5,000$ words simultaneously through 96 independent Attention Heads perfectly ordered by Sine waves, the AI grasps context on a superhuman scale.

Congratulations. You have completed the theoretical architecture of the Transformer. Tomorrow, on **Day 5: Hands-On with Pre-Trained Transformers**, we abandon theory and run `GPT-2` organically using Hugging Face!
