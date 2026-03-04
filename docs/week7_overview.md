# Week 7: Advanced Machine Learning Algorithms

Welcome to Week 7 of the Full-Stack AI Engineer bootcamp! By now, you know how to clean your data and train a solid baseline model like Logistic Regression or a Decision Tree.

But in the world of competitive Data Science (like Kaggle) and high-stakes enterprise AI, a single decision tree is never enough. It's too weak. It easily overfits, or it completely misses complex patterns.

This week, we learn the technique that dominates tabular data worldwide: **Ensemble Learning**. 
Instead of relying on one model, we will train an entire *army* of models and force them to work together.

## What We'll Cover This Week

*   **Day 1: Introduction to Ensembles.** We explore the theory of Wisdom of the Crowds. We learn how algorithms can "vote" to create a prediction far stronger than any individual model could achieve alone.
*   **Day 2: Bagging & Random Forests.** We introduce our first major ensemble strategy: *Bootstrap Aggregating*. We'll build a forest of hundreds of Decision Trees to totally eliminate the variance of a single tree.
*   **Day 3: Boosting & Gradient Boosting.** We learn the polar opposite of Bagging. Instead of training 100 models independently, we train 100 models in a slow sequence, where each new model is specifically trained to fix the mistakes of the previous one!
*   **Day 4: XGBoost.** We introduce `XGBoost` (Extreme Gradient Boosting), the undisputed heavyweight champion of tabular Machine Learning algorithms, known for blazing speed and native regularization.
*   **Day 5: LightGBM & CatBoost.** We step outside the Scikit-Learn ecosystem to explore bleeding-edge algorithms developed by Microsoft and Yandex that can handle massive datasets and raw categorical text natively.
*   **Day 6: Handling Imbalanced Data.** What do you do if your dataset is 99% Normal and 1% Fraud? We introduce `SMOTE` to mathematically synthesize fake minority data so our ensemble models can learn fairly!
*   **Day 7: Ensemble Capstone Project.** We bring everything together by loading the Telco Churn dataset, fixing class imbalance using SMOTE, and running a massive showdown between Random Forest, XGBoost, and LightGBM.

## Why This Matters
If you are working with Images or Text, you use Deep Learning (Neural Networks). But if you are working with CSVs, SQL databases, or Excel spreadsheets (Tabular Data), Neural Networks are often overkill and significantly underperform.

For Tabular Data, **Tree-based Ensemble Models like XGBoost are the undisputed kings.** They are faster, more interpretable, and mathematically superior in almost every business use-case.

Let's build an army. See you tomorrow for **Day 1: Introduction to Ensemble Learning**!
