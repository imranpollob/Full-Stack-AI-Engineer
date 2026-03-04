# Day 5: Creating and Transforming Features

Welcome to Day 5. Sometimes, the data you need doesn't exist yet. Sometimes you have to extract it, derive it, or transform it yourself. This is where Data Scientists become artists.

## 1. Feature Creation (Extraction)
Imagine you are given a dataset of daily bicycle rentals. One column is the Date (`2023-04-12`). Machine Learning algorithms cannot do math on a Date string.

You might think we should delete the column. But wait! The *Day of the Week* (Sunday vs Monday) probably has a massive impact on how many bicycles get rented! 

We can use Pandas to surgically abstract the Day, Month, and Year out of the raw string and turn them into three brand new features! Look at `day5_ex.py`:

```python
# day5_ex.py
import pandas as pd

# Load Bike Sharing Dataset
df = pd.read_csv("bike_sharing_daily.csv")

# 1. Convert the raw text string into a Pandas DateTime object!
df['dteday'] = pd.to_datetime(df['dteday'])

# 2. FEATURE CREATION!
# We extract the hidden underlying data and turn them into brand new columns!
df['day_of_week'] = df['dteday'].dt.day_name() # "Monday", "Tuesday"
df['month'] = df['dteday'].dt.month            # 1 - 12
df['year'] = df['dteday'].dt.year              # 2011, 2012

# (Now we can One-Hot Encode 'day_of_week' and predict bicycle rentals perfectly!)
```

## 2. Feature Transformation (Polynomials)
What if the relationship between the Temperature and Bicycle Rentals isn't a straight line? (People love renting bikes at 70 degrees, but they *hate* renting bikes at 100 degrees).

If you feed the algorithm the flat `temp` feature, it will fail to draw the curve. We can use Scikit-Learn's `PolynomialFeatures` to calculate $x^2$ and $x^3$, handing the model brand new derived features that allow it to map curvature!

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Isolated feature (Temperature) and target (Total count of rentals)
X = df[['temp']]
y = df['cnt']

# Transform the feature into x^2 
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# ... [Train/Test Split happens here] ...

# 1. Train linear model on Original flat data
model_original = LinearRegression()
model_original.fit(X_train, y_train)

# 2. Train linear model on Polynomial Transformed Features
model_poly = LinearRegression()
model_poly.fit(X_poly_train, y_train)

print(f"MSE Original: {mean_squared_error(y_test, model_original.predict(X_test)):.2f}")
print(f"MSE Polynomial: {mean_squared_error(y_test, model_poly.predict(X_poly_test)):.2f}")

# The Polynomial MSE will be significantly lower, because it allowed the 
# model to finally understand the non-linear curvature of summer heat!
```

## Other Transformations
If your data has massive outliers (like income data where one guy is a billionaire), the distribution will be highly "Skewed". A standard transformation trick is to apply a **Logarithmic Transform** (`np.log(X)`). This mathematically pulls massive numbers down, normalizing the distribution into a clean bell curve!

## Wrapping Up Day 5
You have seen how to pull new Features out of raw dates, and how to transform existing data into polynomials and logs.

Tomorrow on **Day 6: Model Evaluation Techniques**, we lock in our understanding of mathematical scoring metrics for predicting models.
