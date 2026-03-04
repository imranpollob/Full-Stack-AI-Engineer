# Day 3: Encoding Categorical Variables

Welcome to Day 3. Machine Learning models only accept numbers. So what do you do if your dataset has a column called `City` with values like `['New York', 'London', 'Tokyo']`?

You must encode the strings into integers. But you must be incredibly careful *how* you encode them, otherwise the AI will misunderstand the math.

## 1. Label Encoding (For Ordinal Data)
Label Encoding simply replaces strings with numbers in alphabetical order.
`Apple = 0`, `Banana = 1`, `Cherry = 2`.

**The Danger:** AI models use math. If Apple is 0 and Cherry is 2, the AI will mathematically assume that *Cherry is 2x larger than Banana*. If your data is "Nominal" (no inherent mathematical order), this ruins the model!

**When to use it:** Only use Label Encoding if your data is "Ordinal" and has a strict hierarchy, like `['Small', 'Medium', 'Large']` -> `[0, 1, 2]`. 

## 2. One-Hot Encoding (For Nominal Data)
To prevent the AI from thinking `London > New York`, we use **One-Hot Encoding**. 
Instead of assigning a number to the text, we create *entirely new columns* (one for each city). We place a `1` if the passenger is from that city, and a `0` if they are not.

By making the cities binary (True/False), they remain mathematically equal!

## Hands-On Let's Encode!
Let's look at `day3_ex.py` using Pandas and Scikit-Learn on the Titanic dataset.

```python
# day3_ex.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load Titanic Dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# 1. ONE-HOT ENCODING (For Nominal Data)
# We don't want the AI thinking "Male" is mathematically greater than "Female".
# We use Pandas get_dummies() to automatically create One-Hot columns!
# drop_first=True prevents the "Dummy Variable Trap" (Multicollinearity).
df_one_hot = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print(df_one_hot.head()) 

# 2. LABEL ENCODING (For Ordinal Data)
# First Class tickets actually ARE "greater" than Third Class tickets. 
# It is safe to Label Encode them!
label_encoder = LabelEncoder()
df['Pclass_encoded'] = label_encoder.fit_transform(df['Pclass'])

# 3. FREQUENCY ENCODING (For High-Cardinality Data)
# The "Ticket" column has 800 unique strings! If we One-Hot encode it, 
# our dataset will explode to 800 columns. 
# Instead, we replace the Ticket String with the Frequency of how often it appeared!
df['Ticket_frequency'] = df['Ticket'].map(df['Ticket'].value_counts())
```

Once all text is banished from the DataFrame, we drop the useless columns (like `Name`), leaving only numbers. The data is finally ready for `model.fit()`!

## Wrapping Up Day 3
Whenever you have Text data:
1.  Is there an order? -> `Label Encoding`
2.  Are there just a few categories? -> `One-Hot Encoding`
3.  Are there hundreds of categories (like Zipcodes)? -> `Frequency / Target Encoding`

Tomorrow on **Day 4: Feature Selection**, we discuss what happens when all this encoding creates *too many* columns!
