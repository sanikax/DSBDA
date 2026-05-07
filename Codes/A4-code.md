# A4 - Data Analytics-1 (Linear Regression)

 

---

# Problem Statement

Create a Linear Regression Model to predict house prices using housing dataset.

---

# 1. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 2. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

---

# 3. Load Inbuilt Dataset

```python
housing = fetch_california_housing()

# Convert into DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add target column
df['Price'] = housing.target
```

---

# 4. Display Dataset

```python
print(df.head())
```

---

# 5. Dataset Information

```python
print("Shape of Dataset:", df.shape)

print("\nColumns:\n", df.columns)

print("\nInformation:")
print(df.info())

print("\nDescription:")
print(df.describe())
```

---

# 6. Check Missing Values

```python
print(df.isnull().sum())
```

---

# 7. Correlation Matrix

```python
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Matrix")

plt.show()
```

---

# 8. Split Data into Training and Testing

```python
X = df.drop('Price', axis=1)

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

---

# 9. Create Linear Regression Model

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

---

# 10. Predict Output

```python
y_pred = model.predict(X_test)
```

---

# 11. Evaluate Model

```python
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

print("R2 Score:", r2_score(y_test, y_pred))
```

---

# 12. Plot Actual vs Predicted Values

```python
plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices")

plt.grid(True)

plt.show()
```

---

# ✅ Conclusion

* Loaded inbuilt housing dataset
* Applied Linear Regression
* Predicted house prices
* Evaluated model using MSE and R² Score

---

 
