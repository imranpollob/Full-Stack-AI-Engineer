# Day 3: Long Short-Term Memory (LSTM) Networks

Welcome to Day 3. Because `SimpleRNN` forgets context due to Vanishing Gradients, researchers invented the **LSTM** (Long Short-Term Memory) cell. 

This single invention revolutionized Machine Translation and Speech Recognition.

## How an LSTM Defeats Amnesia
A `SimpleRNN` blindly overwrites its internal memory on every single time step. 
An LSTM operates fundamentally differently. It introduces a massive "Conveyor Belt" inside the network called the **Cell State**. This conveyor belt runs completely unhindered straight down the entire sequence! Gradients flow flawlessly down the conveyor belt preventing Vanishing Gradients!

But how does data get onto the belt? The LSTM uses an elaborate system of **Gates** to protect the memory.
1.  **The Forget Gate:** Given the new Input, should I completely erase my past memory?
2.  **The Input Gate:** Given the new Input, what new information is important enough to safely write onto the Conveyor Belt?
3.  **The Output Gate:** Based on what is currently stored on the Conveyor Belt, what should my prediction be *right now*?

## Hands-On: LSTM Implementation
Look at `day3_ex.py`. We test the `SimpleRNN` against the `LSTM` side by side!

```python
# day3_ex.py
from tensorflow.keras.layers import SimpleRNN, LSTM

# 1. The Amnesic Pipeline
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# 2. The Conveyor Belt Pipeline!
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    # By changing ONE WORD, we swap the architecture to an LSTM!
    LSTM(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])
```

By substituting `SimpleRNN` with `LSTM`, TensorFlow automatically implements the complex Forget and Input gate mathematics. 
If the movie review starts with *"This movie was horrifying..."*, the LSTM writes "horrifying" onto the Conveyor Belt. Even if the review continues for 500 more words, when the network reaches the final output, "horrifying" is still safely stored on the Cell State, allowing the network to easily predict "Negative Review"!

## Wrapping Up Day 3
If you are processing Sequences, you should almost never use a `SimpleRNN`. Always default to an `LSTM`. 

However, because LSTMs possess 3 different gates, they contain a massive amount of mathematical parameters, making them incredibly slow to train. 

Can we achieve LSTM performance with less computational overhead? Tomorrow, on **Day 4: Gated Recurrent Units (GRUs)**, we introduce Apple's favorite sequence layer.
