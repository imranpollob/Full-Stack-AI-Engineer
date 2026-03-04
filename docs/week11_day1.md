# Day 1: Introduction to Sequence Modeling and RNNs

Welcome to Week 11. Until today, our Neural Networks evaluated data independently. The 10th photograph of a dog didn't rely on the 9th photograph. 

But in **Sequence Modeling**, the current data point is completely dependent on the data point that came before it. If I say the word "United", there is a high mathematical probability the next word is "States".

How do we grant an AI the ability to remember the word "United"? We use a **Recurrent Neural Network (RNN)**.

## The Recurrent Mechanism
A standard Dense layer takes Input $X$, multiplies it by Weights $W$, and produces Output $y$. It forgets $X$ immediately.

A **Recurrent Layer** takes Input $X_{1}$ and produces Output $y_{1}$. BUT, it also saves a copy of its mathematical state (called a **Hidden State**). 
When Input $X_{2}$ arrives, the RNN does not evaluate $X_{2}$ blindly! It combines $X_{2}$ with the Hidden State of $X_{1}$! 

The prediction $y_{2}$ is mathematically influenced by the memory of $X_{1}$!

## Hands-On: IMDB Movie Reviews
Look at `day1_tensorflow.py` and `day1_pytorch.py`. We load the famous IMDB dataset.

```python
# dat1_tensorflow.py
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# 1. Load the top 10,000 most common English words
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Pad the Sequences! (Every movie review must be exactly 200 words long)
max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")

# 3. Build the RNN Pipeline
model = Sequential([
    # Turn English words into mathematical vectors!
    Embedding(input_dim=vocab_size, output_dim=128),
    
    # THE RECURRENT MEMORY LAYER!
    SimpleRNN(128, activation='tanh', return_sequences=False),
    
    # Is the review Positive (1) or Negative (0)?
    Dense(1, activation='sigmoid')
])
```

### What is `pad_sequences`?
A Neural Network requires fixed-size Matrix arrays. But humans write reviews of varying lengths! One review is 15 words; another is 500 words. 
`pad_sequences(maxlen=200)` forces every review to be exactly 200 words long. If it's shorter, it pads the end with invisible `0`s. If it's longer, it brutally truncates it.

## Wrapping Up Day 1
You have built your first "Memory" network utilizing TensorFlow's `SimpleRNN` and PyTorch's `nn.RNN`. It achieved roughly 80% accuracy guessing if a movie review was good or bad!

But how does it train this memory? Backpropagation worked great for static images, but how do we calculate Calculus across the passage of Time?

Tomorrow on **Day 2: Architecture and BPTT**, we explore the mathematical limitations of `SimpleRNN` and why it suffers from "Amnesia".
