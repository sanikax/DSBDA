# A9 - Data Visualization-II

---

# Problem Statement

1. Use the inbuilt dataset `titanic`.
2. Plot a box plot for distribution of age with respect to gender along with survival status.
3. Write observations from the graph.

---

# Pre-requisites

```bash
pip install matplotlib seaborn
```

---

# Code

## 1. Import Libraries

```python
import seaborn as sns
import matplotlib.pyplot as plt
```

---

## 2. Load Inbuilt Dataset

```python
df = sns.load_dataset('titanic')

# Display first 5 rows
df.head()
```

---

## 3. Basic Information About Dataset

```python
print(df.info())
print(df.describe())
```

---

## 4. Check Missing Values

```python
print(df.isnull().sum())
```

---

## 5. Box Plot

```python
plt.figure(figsize=(8,5))

sns.boxplot(
    x='sex',
    y='age',
    hue='survived',
    data=df
)

plt.title('Age Distribution by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')

plt.show()
```

---

# Observations / Inference

1. Female passengers had a higher survival rate compared to males.

2. Most passengers were between age 20 to 40 years.

3. The box plot shows some outliers, meaning a few passengers had very high or very low ages.

4. Male passengers who did not survive are higher in number.

5. The age distribution of females appears slightly more compact compared to males.

---

# Conclusion

The box plot helps visualize the relationship between:

* Gender (`sex`)
* Age (`age`)
* Survival status (`survived`)

It becomes easier to compare survival patterns among male and female passengers.
