# Day 5: Text Preprocessing and Word Embeddings

Welcome to Day 5. Neural Networks are pure math. They require pixel matrices (CNNs) or numerical sequences (RNNs). They throw errors if you feed them the raw string `"Hello"`.

To fix this, we tokenize the dataset. We convert `"The cat sat"` into an array of integers: `[4, 12, 59]`. 

But integers have a fundamental flaw. Is `12` mathematically closer to `4` than it is to `59`? Numbers implicitly have mathematical distances.

If we map `"Cat" = 1` and `"Dog" = 2` and `"Car" = 90`, the numbers incorrectly imply `"Cat"` and `"Dog"` are mathematically related concepts. We need a system that maps semantic meaning directly into numerical Geometry.

We use **Word Embeddings**. 

## What Are Word Embeddings?
Word Embeddings project every single word in a dictionary into a massive 100-Dimensional mathematical coordinate system (a Vector Space). 

The magic? When trained, algorithms dynamically physically push the coordinate of "King" right next to the coordinate of "Queen" because their contextual usage in English is almost identical! "Car" is physically pushed millions of miles away from "Dog".

## Hands-On: GloVe (Global Vectors)
Look at `day5_ex.py`. Instead of forcing our LSTM to learn a brand-new embedding system from scratch on our tiny movie dataset, we download a massive pre-trained Embedding Dictionary called **GloVe**, trained by Stanford on 6 Billion words from Wikipedia!

```python
# day5_ex.py

# 1. Download and map Stanford's Pre-Trained Mathematics into a Dictionary
embedding_index = {}
glove_file = "glove.6B.100d.txt"
with open(glove_file, "r", encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs
        
# 2. Build our Neural Network using the Pre-Trained Dictionary Matrix natively!
model = Sequential([
    Embedding(input_dim=vocab_size, 
              output_dim=100, 
              weights=[embedding_matrix], # Feed it the Stanford Math!
              trainable=False),           # FREEZE THE LAYER! Do NOT alter Stanford's Math!
    
    # Send the flawless 100D coordinates straight to the LSTM!
    LSTM(128, activation='tanh', return_sequences=False),
    
    Dense(1, activation='sigmoid')
])
```

By adding `trainable=False`, we freeze the weights! This is exactly what **Transfer Learning** is. We take Stanford's 6 billion word NLP expertise and hot-swap it directly into our 10,000 word Movie Review Classifier! 

The LSTM instantly performs better because it intrinsically mathematically knows that "Horrible" and "Awful" possess identically negative coordinates.

## Wrapping Up Day 5
You now understand `Embedding` layers. They are the mathematical translators of Language. 

You understand `LSTM` layers. They are the memory engines. 

Tomorrow, on **Day 6: Sequence-to-Sequence (Seq2Seq)**, we combine everything. Instead of returning `False` at the end of the sentence, we force PyTorch to generate an entirely *new sequence*. We are building Google Translate!
