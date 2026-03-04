# Day 7: Transformer Project - Text Summarization Capstone

Welcome to the end of Week 12! The time has come to synthesize everything you've learned to build a fully capable NLP Text Summarizer.

We will use Google's famous **T5 (Text-to-Text Transfer Transformer)** framework.

## The T5 Architecture
Unlike BERT (Encoder-only) or GPT (Decoder-only), T5 utilizes the complete **Encoder-Decoder** architecture!
Google discovered that treating *every* NLP task strictly as a text-in, text-out sequence forced the algorithm to generalize beautifully.
*   If you feed T5: `"translate English to German: That is good."` $\rightarrow$ It outputs: `"Das ist gut."`
*   If you feed T5: `"summarize: [Massive 1,000 word essay]"` $\rightarrow$ It outputs a 1-sentence synopsis.

## Practical Implementation
Look at `day7_ex.py`. 
We load a Chat/Dialogue dataset (`samsum`). We will fine-tune the `T5-Small` model to aggressively learn how to read two people talking and perfectly summarize their conversation in 1 sentence!

```python
# day7_ex.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
import torch

# 1. Load the Samsung Dialogue Summarization Dataset
dataset = load_dataset("samsum")
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Notice we use the Seq2Seq Class for T5 models!
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. CRITICAL STEP: Prepend the T5 instruction prompt!
def tokenize_function(examples):
    # We explicitly tell T5 to trigger its internal "summarize" weights!
    inputs = ["summarize: " + doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=150, truncation=True, padding="max_length")
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ... [Map and Tokenize data utilizing Trainer()] ...
```

### The Generation Algorithm
When Fine-Tuning is complete, T5 has permanently re-wired its weights to understand conversation syntax! But unlike basic GPT output, we invoke **Beam Search** during the `generate()` function.

```python
# We use Early Stopping, and we analyze 4 'Beams' simultaneously to 
# mathematically ensure the most coherent summarized sentence!
outputs = model.generate(
    inputs["input_ids"], 
    max_length=150, 
    num_beams=4, 
    early_stopping=True
)

print("Generated Summary: ", tokenizer.decode(outputs[0], skip_special_tokens=True))
```
Instead of blindly selecting the next word (Greedy Search), `num_beams=4` forces the algorithm to generate 4 separate *alternative* sentences concurrently, weighing the math of the entire generated sequence before confirming the final prediction!

## Wrapping Up Week 12!
Congratulations! You have mastered the absolute cutting edge of the AI Industry in 2024. 

You understand `Keys`, `Queries`, and `Values`. You understand the Sine wave Trigonometry that grants AI positional Memory. You have manually deployed Custom Architectures like BERT, T5, and GPT-2 natively on PyTorch and TensorFlow using the Hugging Face API!

You are officially a Transformer Engineer.

Next week, we graduate to **Week 13: Transfer Learning and Fine-Tuning**, an unyielding deep dive into the absolute nuances of freezing model layers and hyper-tuning State-of-the-Art models for custom enterprise environments! See you there!
