# Week 11: Recurrent Neural Networks (RNNs) and Sequence Modeling

Welcome to Week 11 of the Full-Stack AI Engineer bootcamp! Over the last two weeks, we mastered Deep Learning and Computer Vision. Our algorithms successfully conquered structured 2D images. 

But not all data is a static 2D snapshot. What if your data relies entirely on the passage of time?
*   "The dog bit the man."
*   "The man bit the dog."

These two sentences contain the exact same words (pixels). A Convolutional Network would treat them identically! But the *order* of the words completely changes the meaning. 

To process Language, Audio, and Stock Market data, an algorithm must understand the sequence of time. Welcome to the world of **Sequence Modeling and Recurrent Neural Networks (RNNs)**.

## What We'll Cover This Week

*   **Day 1: Intro to Sequence Modeling.** Why do CNNs fail at reading sentences? We introduce the concept of "Hidden States"—the mathematical mechanism of granting an AI a "Memory".
*   **Day 2: The RNN Architecture & Backpropagation Through Time.** We look under the hood. How does an RNN pass its memory forward? We also explore a fatal mathematical flaw: The Vanishing Gradient Problem. 
*   **Day 3: Long Short-Term Memory (LSTM).** Because basic RNNs suffer from "amnesia", we introduce the LSTM layer. It uses intricate Gates to learn exactly what to remember and what to forget.
*   **Day 4: Gated Recurrent Units (GRUs).** LSTMs are powerful but computationally heavy. We use the GRU, a modernized, lightweight architecture that achieves similar results with fewer parameters.
*   **Day 5: Word Embeddings (GloVe).** An AI cannot read English. We learn how to mathematically map 10,000 English words into a 100-dimensional coordinate system using pre-trained GloVe Embeddings!
*   **Day 6: Sequence-to-Sequence (Seq2Seq).** We build an advanced PyTorch model simulating Google Translate! It will take a sequence of English ("good night") and predict a sequence of French ("bonne nuit").
*   **Day 7: Summary Capstone Project.** We bring it all together by building three separate neural networks (RNN, LSTM, GRU) to classify whether a sequence of 10,000 IMDB Movie Reviews are "Positive" or "Negative".

## Why This Matters
If you want to build Chatbots, automated Language Translators, or predict the future price of Bitcoin based on its past 30 days of trading logic, you *must* master Recurrent Neural Networks. 

Let's give our algorithms a memory. See you tomorrow for **Day 1: Introduction to Sequence Modeling**!
