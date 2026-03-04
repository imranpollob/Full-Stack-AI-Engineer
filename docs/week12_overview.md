# Week 12: Transformers and Attention Mechanisms

Welcome to Week 12 of the Full-Stack AI Engineer bootcamp! Last week, we successfully parsed Language and Sequences using Recurrent Neural Networks (LSTMs).

However, we encountered a massive structural problem: **Sequential Processing**. 
Because an LSTM must read Word 1, before it reads Word 2, before it reads Word 3, it cannot be parallelized. You can own $100$ massive Graphics Cards(GPUs), but $99$ of them will sit idle waiting for the first GPU to finish Word 1. 

In 2017, Google researchers published a paper titled *"Attention Is All You Need"*. They proposed an entirely new architecture that completely eliminated the need for Recurrence (RNNs). They called it the **Transformer**.

## What We'll Cover This Week

*   **Day 1: Intro to Attention.** We dive into the math that killed the RNN. We learn how algorithms calculate `Queries`, `Keys`, and `Values` to dynamically isolate exactly which word in a sentence is the most mathematically important.
*   **Day 2: The Transformer Architecture.** We abandon sequence processing. We learn how Transformers read an entire 5,000-word essay simultaneously, perfectly parallelizing the workload across modern GPUs.
*   **Day 3: Multi-Head Attention.** We realize a sentence has multiple layers of meaning. We learn how to mathematically run 8 "Attention" algorithms concurrently to derive grammar, emotion, and context simultaneously.
*   **Day 4: Positional Encoding.** If a Transformer reads 5,000 words *at the exact same time*, how does it know what order the words were in? We learn the trigonometry hack of injecting Sine and Cosine waves into sentences!
*   **Day 5: Hands-On with BERT and GPT!** We finally touch production algorithms! We download OpenAI's raw `GPT-2` model and generate custom creative writing organically!
*   **Day 6: Using Hugging Face (`Transformers`).** We utilize Python's dominant NLP library to load a $125$-Million Parameter algorithm (`RoBERTa`) and successfully perform Transfer Learning on a News dataset.
*   **Day 7: The Final NLP Capstone.** We build an advanced Sequence-to-Sequence architecture utilizing Google's `T5` Transformer specifically customized to aggressively summarize human conversations. 

## Why This Matters
Everything from ChatGPT to Github CoPilot is built entirely utilizing the architectures we study this week. You are officially touching the absolute bleeding edge of modern AI capability.

Let's dive in. See you tomorrow for **Day 1: Introduction to Attention**!
