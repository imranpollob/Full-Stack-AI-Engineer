# Day 2: Understanding RNN Architecture and BPTT

Welcome to Day 2. Yesterday our RNN successfully classified Movie Reviews. Today, we look at *how* it learned.

In a normal Neural Network, the error propagates backward through the layers (Output $\rightarrow$ Hidden $\rightarrow$ Input). 
In an RNN, the error propagates backward through the layers AND backward through time! (Word 200 $\rightarrow$ Word 199 $\rightarrow$ Word 198...)

This is called **Backpropagation Through Time (BPTT)**.

## The Flaw of the `SimpleRNN`
Look closely at `day2_tensorflow.py`. We run exactly the same code as Day 1.

```python
# day2_tensorflow.py
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

# Output: Test Accuracy: ~0.8200 (82%)
```

Why didn't accuracy reach $99\%$? Because of a fatal mathematical flaw called the **Vanishing Gradient Problem**.

### The Vanishing Gradient Problem
Remember, during BPTT, the Gradients (Calculus error fractions) are multiplied together as they travel backward through time. 
If you multiply $0.5 \times 0.5 \times 0.5$, the number shrinks extremely fast! 

By the time the algorithm propagates the error from Word #200 all the way back to Word #1, the Gradient has functionally vanished to $0.000000001$. 

### What does this mean in real life?
It means the `SimpleRNN` has **terrible short-term memory**. 

If a movie review starts with: *"I absolutely loved this movie..."*, but then goes on a 150-word tangent about the popcorn at the theater... by the time the algorithm reaches word #200 to make its final prediction, the Vanishing Gradient has caused it to mathematically forget that the first word was "loved"!

It suffers from profound amnesia when forced to read long sequences. 

## Wrapping Up Day 2
Vanilla/Simple RNNs cannot be used in Deep Learning if your sequence is longer than ~15 data points. They are mathematically incapable of connecting a data point at $T=0$ with a data point at $T=100$.

To solve natural language processing (where paragraphs are thousands of words long), researchers had to invent a completely new type of Recurrent block that was immune to the Vanishing Gradient.

Tomorrow, on **Day 3: Long Short-Term Memory (LSTM)**, we replace the `SimpleRNN` with the architecture that defined NLP for a decade.
