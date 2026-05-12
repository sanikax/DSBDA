# A10 - Data Visualization-III

---

# Pre-requisites

Install required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

# Code

## 1. Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
```

---

## 2. Load Inbuilt Iris Dataset

```python
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['variety'] = iris.target

# Convert target values into flower names
df['variety'] = df['variety'].map({
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
})

# Display first 5 rows
df.head()
```

---

# Problem Statement Tasks

---

## 3. Features and Their Data Types

```python
print("Features and their Data Types:\n")
print(df.dtypes)
```

---

## 4. Histogram for Each Feature

```python
plt.figure(figsize=(10,8))

for i, column in enumerate(df.columns[:-1]):

    plt.subplot(2,2,i+1)

    plt.hist(df[column], bins=10, edgecolor='black')

    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
```

---

## 5. Box Plot for Each Feature

```python
plt.figure(figsize=(10,8))

for i, column in enumerate(df.columns[:-1]):

    plt.subplot(2,2,i+1)

    sns.boxplot(x=df[column])

    plt.title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()
```

---

## 6. Detect Outliers and Compare Distributions

```python
for column in df.columns[:-1]:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)][column]

    print("\nFeature:", column)

    print("Mean =", round(df[column].mean(),2))
    print("Median =", round(df[column].median(),2))
    print("Standard Deviation =", round(df[column].std(),2))

    if len(outliers) > 0:
        print("Outliers Present")
        print("Outlier Values:", outliers.tolist())

    else:
        print("No Outliers")

    print("-"*40)
```

---

# Inference

* Histograms show the distribution of features.
* Boxplots help identify outliers.
* Mean, median and standard deviation help compare distributions.
* Outliers are detected using the IQR method.

---
 
