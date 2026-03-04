# Day 5: Hands-On with Pre-Trained Transformers

Welcome to Day 5. The theoretical architecture is complete. Now, we use Python's legendary `transformers` library (built by **Hugging Face**) to access the most powerful Open-Source AI algorithms in human history.

Hugging Face allows you to download $1.5$ Billion Parameter Transformer algorithms directly to your computer using 3 lines of code! 

## OpenAI's GPT
Let's download OpenAI's revolutionary early Transformer: `GPT-2`.

Unlike BERT (an Encoder designed for classification), `GPT-2` is a pure **Decoder**. It is a Causal Language Model. Its sole architecture revolves around reading your *"Prompt"* and predicting the most mathematically probable next word.

Once it predicts that word, it adds that word back to the prompt, and predicts the *next* next word! (An Auto-Regressive loop). 

## Hands-On: Causal Text Generation
Look closely at `day5_ex.py`. 

```python
# day5_ex.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Hugging Face automatically downloads the Model Weights and Token Dictionary!
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. We write a Prompt!
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 3. Tell the model which tokens are real, and which are padding `<PAD>`
attention_mask = input_ids.ne(tokenizer.pad_token_id)
pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1

# 4. EXECUTE THE GENERATOR LOOP! (Auto-Regressive generation!)
output = gpt_model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=50,             # Keep generating until 50 words!
    num_return_sequences=1,
    pad_token_id=pad_token_id
)

print("Generated Text:", tokenizer.decode(output[0], skip_special_tokens=True))

# Sample Output: "Once upon a time, I was a very different kind of person. I was a very happy, outgoing, and deeply spiritual person..."
```

### The Power of Hugging Face
Look at the `generate()` parameters.
You control the absolute structure of the AI! You can change `max_length`. You can increase the "creativity" or randomness by adjusting parameters like `temperature`. You can force it to generate 5 completely separate stories simultaneously using `num_return_sequences=5`.

You have just run a Generative AI algorithm completely locally on your own machine. 

## Wrapping Up Day 5
OpenAI's GPT models are foundational text engines. But they are completely generalized. They know a little bit about everything on the internet.

What if we want an incredibly specific, hyper-focused algorithm designed *solely* for classifying legal documents? We would have to **Fine-Tune** a pre-trained Transformer! 

Tomorrow, on **Day 6: Advanced Transformers**, we download a $125M$ parameter algorithm named `RoBERTa` and use Transfer Learning to permanently alter its internal matrices to classify the news!
