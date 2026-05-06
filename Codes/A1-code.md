# A1 - Data Wrangling - I
---

## Pre-requisites

```bash
pip install pandas numpy scikit-learn
```

---

## Code

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
```

---

### 2. Load Inbuilt Dataset

```python
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column
df['variety'] = iris.target
```

---

### 3. Dataset Description

```python
print(iris.DESCR)
```

---

### 4. Basic Exploration

```python
print(df.head())
print(df.tail())
print(df.shape)
```

---

### 5. Data Preprocessing

```python
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
```

---

### 6. Data Formatting

```python
print(df.dtypes)
```

(All columns are already numeric in this dataset)

---

### 7. Convert Numerical → Categorical (for understanding)

```python
df['variety'] = df['variety'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})
```

---

### 8. Convert Categorical → Numerical (as required)

```python
df['variety'] = df['variety'].map({
    'Setosa': 0,
    'Versicolor': 1,
    'Virginica': 2
})
```

---

### 9. Data Normalization

```python
numeric_cols = df.select_dtypes(include=np.number)

df[numeric_cols.columns] = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())
```

---

### 10. Final Output

```python
print(df.head())
```

---

## ✅ Summary

* Used inbuilt dataset (no file dependency)
* Checked missing values and duplicates
* Verified data types
* Performed categorical conversion
* Applied normalization

---