# Day 4: Gated Recurrent Units (GRUs)

Welcome to Day 4. Yesterday we looked at LSTMs. LSTMs are brilliant at completely solving the Vanishing Gradient problem and remembering long context using 3 unique mathematical Gates.

But here is the problem: Those 3 Gates require 3 unique massive dense matrix multiplications. 
If you train a 100-layer LSTM over millions of sentences, the Training time expands drastically.

Can we achieve the exact same "Conveyor Belt" memory, but combine the Gates so the math is cut by 30%? We use a **Gated Recurrent Unit (GRU)**.

## How a GRU Works
The GRU merges the LSTM's `Forget` and `Input` Gates into a single `Update Gate`!

1.  **The Update Gate:** Does the new word contain critical information? If yes, overwrite the Conveyor belt! If no, completely ignore it.
2.  **The Reset Gate:** Based on the new word, what past information from the Conveyor belt do I need to *erase right now*?

Because it only uses 2 Gates instead of 3, the parameter count drops massively, drastically reducing RAM overhead and vastly speeding up the training pipeline without sacrificing long-term memory!

## The Keras Implementation
Look at `day_ex4.py`. We run exactly the same code as yesterday, but we initialize a `GRU`!
```python
# day_ex4.py
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU

# 1. The Amnesic Pipeline
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# 2. The Heavy Conveyor Belt Pipeline!
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# 3. The Lightweight Conveyor Belt Pipeline!
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])
```

## GRU vs LSTM
If GRUs are faster and use less RAM, why wouldn't we use them all the time?
1.  **LSTM:** Because they use more parameters natively, they are historically strictly better at incredibly dense, incredibly long sequences where the subtle nuances of 3 distinct gates are superior at parsing complex linguistic contexts.
2.  **GRU:** Usually the default starting point for engineers. GRUs are much faster to train on smaller datasets and perform identically to LSTMs in 90% of Time-Series and basic NLP applications.

## Wrapping Up Day 4
You now have the Holy Trinity of Recurrent Layers in your arsenal: `RNN`, `LSTM`, and `GRU`.

But wait. Have you noticed the mysterious `Embedding` layer at the start of all our networks? Why couldn't we just pass raw text strings directly into the LSTM? 

Because Neural Networks cannot read English. Tomorrow, on **Day 5: Word Embeddings**, we learn exactly how scientists convert the entire English Dictionary into mathematical coordinates using `GloVe`.
