# Day 4: Transfer Learning in NLP

Welcome to Day 4. Today, we leave Computer Vision behind and dive into the architecture that disrupted the tech industry in 2018: **BERT**.

Last week, you learned the mathematical underpinnings of the Transformer architecture. You now understand how Attention blocks operate, and you understand `Positional Encoding`. 

Today, we use **Transfer Learning** to hijack Google's foundational NLP engine and hot-swap its mathematical output to solve *our* specific English language tasks!

## BERT vs GPT vs T5
There are three fundamental types of pre-trained NLP models.
1.  **BERT (Encoder-Only):** Trains by reading millions of Wikipedia articles and aggressively randomly masking (hiding) $15\%$ of the words. It forces its own math to perfectly guess the masked word by reading the text *before* and *after* the invisible word. Because it looks symmetrically backward and forward at the exact same millisecond, it possesses an unbelievably robust understanding of context. Used for Classification.
2.  **GPT (Decoder-Only):** Trains by reading millions of internet articles. Its sole objective is to logically deduce the *very next word* at the end of the sentence. Used for Text Generation.
3.  **T5 (Encoder-Decoder):** Takes text-in, generates text-out. Because it treats everything natively as a generation task, it is the absolute industry standard for Translation and Summarization.

## Hands-On: Hugging Face Trainer
Look at `day4_ex.py`. We download the foundational `bert-base-uncased` and fine-tune it in 10 lines of code.

```python
# day4_ex.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Download IMDB (50,000 movie reviews)
dataset = load_dataset("imdb")

# 2. Download Google's pre-trained vocabulary token mapper!
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3. Download Google's 110-Million parameter algorithm, but REPLACE THE HEAD with 2 Labels!
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# ... [Execute Tokenizer Mapping] ...

# 4. INITIALIZE THE AUTOMATED TUNING ENGINE!
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,            # Tiny Learning Rate! (Don't destroy BERT's English grammar!)
    per_device_train_batch_size=16,
    num_train_epochs=1,            # 1 single epoch is enough!
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
```

### The NLP Tokenizer
Computer Vision assumes every pixel is a number from $0$ to $255$. 
Language has no standard numbers. You must download the exact `Tokenizer` that matches the exact `Model` you are fine-tuning. 

`bert-base-uncased` exclusively understands WordPiece tokens. If you accidentally feed it text tokenized using GPT's Byte-Pair Encoding (BPE), the matrix will mathematically collapse. The AI will output junk! 

## Wrapping Up Day 4
You have just orchestrated a complete, industry-standard NLP Fine-Tuning pipeline using the Hugging Face `Trainer`! 

But what if you want tighter control over the learning loop? The `Trainer` abstracts everything away into a black box. What if you want to implement advanced PyTorch schedulers? 

Tomorrow on **Day 5: Fine-Tuning Techniques in NLP**, we rewrite the Trainer manually using Raw PyTorch logic to integrate Discriminative Fine-Tuning and Slanted Triangular Learning Rates (STLR)!
