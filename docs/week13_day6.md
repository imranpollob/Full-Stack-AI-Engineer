# Day 6: Domain Adaptation and NLP Challenges

Welcome to Day 6. So far, we've fine-tuned BERT on IMDB movie reviews and AG News articles. 
Because BERT was trained on Wikipedia, its internal coordinate system fundamentally understands common English syntax, making classification trivial.

What happens if you try to fine-tune `bert-base-uncased` to classify highly technical Medical Oncology reports? 
The model will completely collapse!

Why? **Data Mismatch**.
BERT's vocabulary does not contain the word `Carcinoma`. It does not contain `Hepatotoxicity`. Therefore, the tokenizer breaks the string down into meaningless garbage subsets, shattering the context matrix. 

## The Solution: Domain Adaptation
To fix Data Mismatch phenomena, we execute **Domain Adaptation**. 
Instead of starting from a generalized state (Wikipedia), we explicitly hunt down a pre-trained model that has already spent thousands of hours specifically reading Medical documents! 

Researchers aggressively fork standard models into specialized Domains:
*   **BioBERT:** Pre-trained exclusively on `PubMed` (medical articles).
*   **LegalBERT:** Pre-trained exclusively on European Legal contracts and statutes.
*   **SciBERT:** Pre-trained on semantic scientific research papers.

## Hands-On: The BioBERT Architecture
Look at `day6_ex.py`. We utilize the `pubmed_rct` (Randomized Controlled Trials) medical dataset! If we tried to train this with `bert-base-uncased`, we would get mediocre F1-Scores.

Instead, we hot-swap the foundational `repo_id` inside Hugging Face immediately!
```python
# day6_ex.py
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

# 1. Load the RCT Medical Dataset!
dataset = load_dataset("pubmed_rct", "20k_rct")

# 2. Download a Domain-Specific Tokenizer tailored for Medical Vocabulary!
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# 3. DO NOT IMPORT BERT! Import the BioBERT Medical Matrix!
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-base-cased-v1.1",
    num_labels=5
)

# ... [Execute processing via Trainer!] ...
```

By switching 2 strings (`"bert-base-uncased"` to `"dmis-lab/biobert-base-cased-v1.1"`), we instantly boost our baseline accuracy by leaps and bounds. The model physically *knows* what a `tumor` is, whereas generalized BERT treats `tumor` as an unknown outlier.

### Text Data Augmentation
Look at the bottom of `day6_ex.py`. Unlike Images (where you can just dynamically rotate the picture), you cannot arbitrarily flip or rotate English text!
To organically create more robust data to aid Domain Adaptation, we programmatically employ Synonym Replacement!

```python
import random

def augment_text(text):
    synonyms = {"cancer": ["tumor", "malignancy"], "study": ["research", "experiment"]}
    words = text.split()
    # Programmatically augment the string!
    new_words = [random.choice(synonyms[word]) if word in synonyms else word for word in words]
    return " ".join(new_words)
```
By organically introducing synthetic terminology into the text corpus, the fine-tuning mechanism is forced to generalize the context array! 

## Wrapping Up Day 6
If your custom task involves highly technical documentation, Code, Legal Text, or Medicine, you must employ Domain Adaptation. Seek out specialized backbone models on the Hugging Face Hub (e.g., `CodeLlama`, `LegalBERT`) rather than blindly relying on general architecture!

Tomorrow, on **Day 7: The Final Transfer Capstone**, we orchestrate a completely definitive image transfer learning pipeline! We tie Computer Vision models together, locking all frozen architectures into a fully implemented PyTorch validation system!
