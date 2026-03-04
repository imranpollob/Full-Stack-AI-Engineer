# Day 2: Introduction to Transformers Architecture

Welcome to Day 2. Yesterday we learned about Attention—how words calculate their semantic importance relative to every other word in a sentence.

Now, we zoom out to the macro-level: **The Transformer Architecture**. 
The Transformer is the physical chassis that houses the Attention engine. 

## The Core Concept: Parallelization
LSTMs read text sequentially. The Transformer reads everything *simultaneously*.
If you pass a 5,000-word text into a Transformer, all 5,000 words hit the Attention mechanism at the exact same millisecond. 
Because mathematical matrix operations run blazing fast on modern GPUs, a Transformer can train on millions of documents in the time it takes an LSTM to train on a few thousand.

## Encoder vs Decoder
Like the Seq2Seq models from last week, the Transformer is divided into two halves:

### 1. The Encoder (Understanding)
The Encoder's sole job is to process an Input sequence and create massive mathematical context embeddings using Attention.
*   **Famous Model:** **BERT** (Bidirectional Encoder Representations from Transformers). 
*   **Use Case:** BERT is used when the AI needs a profound *understanding* of what it is reading (e.g., Sentiment classification, Question Answering, Google Search). It reads "Bidirectionally"—meaning to understand the word "bank", it looks at the words before it *and* after it simultaneously.

### 2. The Decoder (Generating)
The Decoder's sole job is to take Context, and predict the *very next word* sequentially.
*   **Famous Model:** **GPT** (Generative Pre-Trained Transformer).
*   **Use Case:** GPT is used for Creative text generation. Unlike BERT, GPT only reads "Unidirectionally". It can only look at the words that came *before* it, forcing it to guess the future!

## Hands-On: Invoking the Encoder (BERT)
Look at `day2_ex1.py`. We download the official `BERT` transformer using Python's famous `transformers` package!

```python
# day2_ex1.py
from transformers import BertTokenizer, BertModel

# 1. Download Google's pre-trained tokenizer mapping English to BERT's internal math
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. Download the 110-Million Parameter Transformer!
model = BertModel.from_pretrained("bert-base-uncased")

# 3. Tokenize our Sentence!
text = "Transformers are powerful models for NLP tasks"
inputs = tokenizer(text, return_tensors='pt')

# 4. Pass the sequence into BERT!
outputs = model(**inputs)

print("Hidden States Shape:", outputs.last_hidden_state.shape)
# Output: [1, 9, 768]
```

### Deciphering the Output
Look at the output shape: `[1, 9, 768]`. 
*   `1`: We passed in 1 sentence.
*   `9`: The sentence contained 7 words + 2 hidden control tokens (`<CLS>` and `<SEP>`).
*   `768`: **The Magic.** BERT mathematically expanded the context of *every single word* into 768 distinct dimensions of understanding!

## Wrapping Up Day 2
Google's BERT is an Encoder architecture. OpenAI's GPT is a Decoder architecture. 

But wait, look at the size of BERT's output: $768$ dimensions of semantic understanding for a single word. How does an Attention block possibly track $768$ simultaneous concepts?

Tomorrow, on **Day 3: Multi-Head Attention**, we mathematically split the Attention algorithm into cloned pieces.
