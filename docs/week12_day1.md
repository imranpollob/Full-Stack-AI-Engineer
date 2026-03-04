# Day 1: Introduction to Attention Mechanisms

Welcome to Week 12. Before we can build a Transformer, we must understand the algorithm that makes it possible: **The Attention Mechanism**.

In an LSTM context, meaning is passed linearly down a conveyor belt. If a sentence is 5,000 words long, the final "Context Vector" is just a massive blur of information. 

The Attention mechanism solves this by dynamically calculating *which specific words matter the most* relative to the current word being processed!

## Queries, Keys, and Values
Let's imagine the AI is translating: `"The quick brown fox jumped over the lazy dog."`

When the AI is focused on translating the word `"jumped"`, which other words in that sentence should it pay the most *attention* to? Probably `"fox"` (who jumped?) and `"over"` (where did it jump?). It shouldn't pay any attention to `"lazy"`.

To calculate this mathematically, the algorithm creates 3 separate matrices for every single word:
1.  **Query ($Q$):** "I am the word *jumped*. I am looking for a noun that caused me." (What the word is searching for).
2.  **Key ($K$):** "I am the word *fox*. I am a noun that performs actions." (What the word structurally represents).
3.  **Value ($V$):** The actual mathematical English meaning of the word itself.

## The Attention Math
Look at `day1_numpy.py` and `day1_pytorch.py`. We calculate this manually!

```python
# day1_pytorch.py
import torch
import torch.nn.functional as F

# Assume these are 3-Dimensional embeddings for a sentence containing 3 words.
queries = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
keys = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
values = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

# 1. The Dot Product calculates the semantic similarity between the Query and the Keys!
scores = torch.matmul(queries, keys.T)

# 2. We use Softmax to perfectly convert those scores into percentages that equal 100%!
attention_weights = F.softmax(scores, dim=-1)

# 3. We multiply the semantic Values by the attention weights! 
context = torch.matmul(attention_weights, values)

print("Attention Weights:\n", attention_weights)
```

### The Magic of the Dot Product
Because Query and Keys are both vectors, calculating the Dot Product of the matrices determines the exact mathematical angle between them! 
If the Query ("jumped") has a perfectly aligned angle with the Key of ("fox"), the Dot Product returns an incredibly high score! 

When passed through the `Softmax` filter, the algorithm will confidently dictate that `"jumped"` must assign $95\%$ of its Attention to `"fox"`.

## Wrapping Up Day 1
Instead of generating a blurry single context vector at the end of a sentence like an LSTM, the Attention Mechanism calculates the semantic relationship of *every single word against every other word in the sequence*. 

Tomorrow on **Day 2: The Transformer architecture**, we will take the Attention score algorithm and build the framework that dethroned RNNs.
