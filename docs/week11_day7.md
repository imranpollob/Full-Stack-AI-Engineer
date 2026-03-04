# Day 7: RNN Project & Sentiment Analysis Capstone

Welcome to the end of Week 11! The time has come to synthesize everything you've learned. You understand recurrent Hidden States, the mathematical complexity of Embeddings, the Vanishing Gradient problem, and the Conveyor Belt mechanism of LSTMs and GRUs.

Today, we execute a definitive performance shootout on the **IMDB Sentiment Analysis Dataset**.

## The Architecture Shootout
Look at `day7_project.py`. 
We build three identical Keras pipelines side-by-side. The *only* difference natively is the Recurrent Layer! We train all three simultaneously and evaluate the final Test Accuracy.

```python
# day7_project.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense

# Pipeline 1: The Vanilla SimpleRNN
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Pipeline 2: The mathematically heavy LSTM!
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Pipeline 3: The Optimized Lightweight GRU!
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# ... [Compile, fit, evaluate logic using Adam] ...
```

### Output Results
If you run `day7_project.py`, PyPlot dynamically charts the Training Accuracy over Epochs! 
1.  **SimpleRNN (`0.8124`):** The network suffers heavily from Vanishing Gradients right at the start. It successfully parses short movie reviews but totally misidentifies paragraphs.
2.  **LSTM (`0.8643`):** The Conveyor belt flawlessly passes Sentiment logic across the sequence regardless of length! But note the processing time natively is much slower!
3.  **GRU (`0.8690`):** The 2-Gate architecture performs nearly identically to the LSTM engine, but computationally runs 30% faster!

## Wrapping Up Week 11!
Congratulations! You have mastered Sequence Modeling. You understand exactly why `SimpleRNNs` are historically obsolete. You understand `LSTMs` and `GRUs` dominate time-series contexts.

But I have a terrible secret to tell you.

Just like `Dense` networks became entirely obsolete for images (replaced by `CNNs`), Recurrent mechanisms like `LSTM` and `GRU` have now become completely obsolete for Natural Language Processing!

Why? Because an LSTM *must* read a 500-word paragraph sequentially (word 1, then word 2, then word 3). This means you cannot effectively multi-thread processing across modern GPUs. LSTMs are unbelievably slow to train on big datasets.

To achieve Chat-GPT levels of performance, researchers invented an architecture that completely abandons sequential processing. It reads all 500 words of a sequence *simultaneously*!

Next week, we graduate to **Week 12: Transformers and Attention Mechanisms**. Welcome to the Modern AI Era. See you there!
