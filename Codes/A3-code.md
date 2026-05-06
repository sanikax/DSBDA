# A3 - Descriptive Statistics 
 
---

## 🔹 Part 1: Student Dataset

### 1. Import Libraries

```python
import pandas as pd
import numpy as np
```

---

### 2. Generate Dataset

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
print(df.head())
```

---

### 3. Create Categorical Group (Marks Group)

```python
df['Marks_Group'] = pd.cut(df['Marks'],
                          bins=[0,50,70,100],
                          labels=['Low','Medium','High'])
```

---

### 4. Summary Statistics (Grouped)

```python
group_stats = df.groupby('Marks_Group')['Marks'].describe()
print(group_stats)
```

✔ Includes mean, std, min, max, percentiles

---

### 5. Median

```python
median_marks = df.groupby('Marks_Group')['Marks'].median()
print("Median Marks:\n", median_marks)
```

---

### 6. Mode

```python
print("Mode Age:", df['Age'].mode().values)
print("Mode Marks:", df['Marks'].mode().values)
```

---

### 7. Mean, Std, Min, Max

```python
print("Mean:\n", df.groupby('Marks_Group')['Marks'].mean())
print("Std:\n", df.groupby('Marks_Group')['Marks'].std())
print("Min:\n", df.groupby('Marks_Group')['Marks'].min())
print("Max:\n", df.groupby('Marks_Group')['Marks'].max())
```

---

## 🔹 Part 2: Inbuilt Iris Dataset

### 8. Load Dataset

```python
from sklearn.datasets import load_iris

iris = load_iris()

df2 = pd.DataFrame(iris.data, columns=iris.feature_names)
df2['species'] = iris.target
```

---

### 9. Convert Species

```python
df2['species'] = df2['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})
```

---

### 10. Summary Statistics

```python
print(df2.groupby('species').describe())
```

---

### 11. Percentiles

```python
print(df2.groupby('species').quantile([0.25, 0.5, 0.75]))
```

---

### 12. Mean, Std, Min, Max

```python
print("Mean:\n", df2.groupby('species').mean())
print("Std:\n", df2.groupby('species').std())
print("Min:\n", df2.groupby('species').min())
print("Max:\n", df2.groupby('species').max())
```

---

### 13. Median

```python
print("Median:\n", df2.groupby('species').median())
```

---

### 14. Mode Example

```python
print("Mode:", df2['sepal width (cm)'].mode().values)
```

---

## ✅ Conclusion

* Random student dataset generated
* Grouped statistics calculated using Marks_Group
* Mean, median, mode, std, percentiles computed
* Iris dataset analyzed using inbuilt data

---