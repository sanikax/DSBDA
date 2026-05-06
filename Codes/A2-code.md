# A2 - Data Wrangling (Academic Performance Dataset)

## Aim

Create an academic performance dataset and perform:

* Missing value handling
* Outlier detection & handling
* Data transformation

---

## Import Libraries

```python
import pandas as pd
import numpy as np
```

---

## Create Dataset

```python
np.random.seed(10)

data = {
    'Student_id': range(1, 51),
    'Age': np.random.randint(18, 25, 50),
    'Attendance': np.random.randint(60, 100, 50),
    'Marks': np.random.randint(40, 100, 50),
    'Gender': np.random.choice(['Male', 'Female'], 50)
}

df = pd.DataFrame(data)
df.head()
```

---

## Introduce Missing & Inconsistent Values

```python
df.loc[5, 'Age'] = np.nan
df.loc[10, 'Marks'] = np.nan
df.loc[8, 'Attendance'] = 150  # invalid value

df.head(15)
```

---

## Check Missing Values

```python
df.isnull().sum()
```

---

## Handle Missing Values

```python
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())
```

---

## Handle Inconsistent Data

```python
df['Attendance'] = df['Attendance'].apply(lambda x: 100 if x > 100 else x)
```

---

## Detect Outliers (IQR Method)

```python
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df[(df['Age'] < lower) | (df['Age'] > upper)]
```

---

## Handle Outliers

```python
df['Age'] = np.where(df['Age'] > upper, upper,
             np.where(df['Age'] < lower, lower, df['Age']))
```

---

## Data Transformation (Normalization)

```python
df['Scaled_Marks'] = (df['Marks'] - df['Marks'].min()) / (df['Marks'].max() - df['Marks'].min())

df[['Marks', 'Scaled_Marks']].head()
```

---

## Final Output

```python
df.head()
```

---

## Conclusion

* Missing values handled using mean
* Invalid data corrected
* Outliers detected using IQR method
* Normalization applied to Marks
