# Day 6: Advanced Transformers - Fine-Tuning BERT Variants

Welcome to Day 6. Yesterday, we used `GPT-2` for General Text Generation. 
But generative text is notoriously bad for strict analytical tasks (like Classifying $10,000$ legal documents accurately). Generative models hallucinate. 

For strict, deterministic classification, we return to the **Encoder (BERT)** framework.

## The BERT Variants
The original BERT was revolutionary, but Researchers quickly discovered inefficiencies. They built optimized variations:
1.  **RoBERTa (Robustly Optimized BERT):** Stripped out the "Next Sentence Prediction" phase of BERT, using massive batch sizes and more data to drastically outperform original BERT on classification tasks.
2.  **DistilBERT:** Smashed the architecture down, retaining $97\%$ of BERT's performance while running $60\%$ faster! Ideal for mobile phones or low-RAM environments.
3.  **ALBERT (A Lite BERT):** Shares mathematical parameters across layers to drastically shrink memory consumption!

## Hands-On: Transfer Learning with RoBERTa
Look at `day6_ex.py`. We load the massive `ag_news` dataset from Hugging Face containing thousands of articles. 
We want to Fine-Tune `RoBERTa` to categorize these articles automatically into $4$ labels: (World, Sports, Business, Sci/Tech).

Instead of PyTorch's complex boilerplate loops, Hugging Face provides the `Trainer` API to streamline the entire Transfer Learning pipeline!

```python
# day6_ex.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Download the AG News Dataset (120,000 Articles!)
dataset = load_dataset("ag_news")

# 2. Download Facebook/Meta's Pre-trained RoBERTa!
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Note we specify 4 Labels! We are overwriting RoBERTa's output layer!
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
# ... [Format keys to float tensors and remove 'text' columns]

# 3. Setup the High-Performance Fine-Tuning Configuration!
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,       # TINY Learning Rate! Do NOT destroy RoBERTa's base math!
    per_device_train_batch_size=16,
    num_train_epochs=3,       # Transfer Learning is blazing fast! Only 3 epochs!
    weight_decay=0.01         # Aggressive Regularization to prevent Overfitting!
)

# 4. Initialize Hugging Face's automated Trainer!
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer
)

# EXECUTE FINE-TUNING ALGORITHM!
trainer.train()

# Output Results: Test Accuracy > ~94%!
```

## Wrapping Up Day 6
With exactly $50$ lines of Python, you downloaded a $125+$ Million Parameter State-of-the-Art Language Engine trained by Meta's Supercomputers... and perfectly re-wired it to blindly classify World News categories at superhuman accuracy in under $20$ minutes!

This is the power of the `transformers` library. 

Tomorrow, on **Day 7: The Final NLP Capstone**, we move past simple Classification and Generative prompting. We will utilize Google's **Text-To-Text Transfer Transformer (T5)** to aggressively auto-summarize a dataset containing thousands of human conversations!
