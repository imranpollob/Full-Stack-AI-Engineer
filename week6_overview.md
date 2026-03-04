# Week 6: Feature Engineering and Model Evaluation

Welcome to Week 6 of the Full-Stack AI Engineer bootcamp! Last week, we successfully trained our first machine learning algorithms. We threw data at Scikit-Learn, and it mathematically figured out the answers.

But in the real world, data is rarely ready for an algorithm. It's filled with missing values, text strings instead of numbers, and wildly different scales (like comparing Age in years to Salary in dollars). If you feed garbage into a machine learning model, you get garbage out.

This week is dedicated to the dark art that occupies 80% of a Data Scientist's actual time: **Feature Engineering**.

## What We'll Cover This Week

*   **Day 1: Introduction to Feature Engineering.** We define what a "Feature" is in the real world, and how to identify categorical vs. numerical data using Pandas.
*   **Day 2: Data Scaling & Normalization.** Distance-based algorithms (like k-NN) fail spectacularly if data isn't scaled. We learn how to mathematically squash all data down to the exact same size using `MinMaxScaler` and `StandardScaler`.
*   **Day 3: Encoding Categorical Variables.** Mathematics cannot multiply the word "Male" by 5. We must convert text to numbers using `OneHotEncoder` and `LabelEncoder`.
*   **Day 4: Feature Selection.** If you have 500 columns of data, your model will be slow and overfit. We learn how to systematically delete useless columns using Correlation Matrices and Tree-based Feature Importance.
*   **Day 5: Creating & Transforming Features.** Sometimes the data you need is hiding. We learn to extract the "Day of the Week" from a raw date string, and bend straight lines using `PolynomialFeatures`.
*   **Day 6: Advanced Model Evaluation.** We revisit accuracy metrics and deeply explore when to use MAE vs MSE for Regression, and when to use Precision vs Recall for Classification.
*   **Day 7: Cross-Validation & Hyperparameter Tuning.** We put everything together to build a robust model using `K-Fold` validation, and introduce the concept of `GridSearch` to autonomously find the absolute best settings for an algorithm.

## Why This Matters
As the saying goes: *"A simple algorithm with great features will always beat a complex algorithm with terrible features."*

Neural networks get all the glory, but Feature Engineering is where the actual problems are solved. 

Let's learn how to mold data. See you tomorrow for **Day 1: Introduction to Feature Engineering**!
