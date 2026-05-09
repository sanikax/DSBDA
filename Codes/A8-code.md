# A8 - Data Visualization-1
 

---

## Problem Statement

Use the inbuilt dataset **'titanic'** from Seaborn library.
Perform data visualization to find patterns in the dataset.
Plot a histogram to check how the ticket fare (`fare`) is distributed among passengers.

---

## Pre-requisites

```bash
pip install matplotlib seaborn pandas
```

---

# Code

## 1. Import Libraries

```python3
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 2. Load Inbuilt Titanic Dataset

```python3
df = sns.load_dataset('titanic')

print(df.head())
```

---

## 3. Check Dataset Information

```python3
print(df.info())
print(df.describe())
```

---

## 4. Histogram of Fare Distribution

```python3
plt.figure(figsize=(6,4))

sns.histplot(data=df, x='fare', bins=10)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.show()
```

---

## 5. Box Plot

```python3
plt.figure(figsize=(6,4))

sns.boxplot(x='class', y='fare', data=df)

plt.title("Fare by Passenger Class")

plt.show()
```

---

## 6. Scatter Plot

```python3
plt.figure(figsize=(6,4))

sns.scatterplot(x='age', y='fare', data=df)

plt.title("Age vs Fare")

plt.show()
```

---

## 7. Violin Plot

```python3
plt.figure(figsize=(6,4))

sns.violinplot(x='class', y='age', data=df)

plt.title("Age Distribution by Class")

plt.show()
```

---

## ✅ Summary

* Loaded Titanic inbuilt dataset
* Visualized fare distribution using histogram
* Used box plot, scatter plot, and violin plot to identify patterns in the data
---
