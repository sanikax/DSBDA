# A6 - Data Analytics-3

---

# Problem Statement

Implement Simple Naïve Bayes Classification algorithm using Python on Iris dataset.

Compute:

* Confusion Matrix
* TP, TN, FP, FN
* Accuracy
* Error Rate
* Precision
* Recall

---

# Pre-requisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# Code

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
```

---

## 2. Load Inbuilt Iris Dataset

```python
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['variety'] = iris.target

print(df.head())
```

---

## 3. Set Independent and Dependent Variables

```python
X = df.drop('variety', axis=1)   # Independent variables
y = df['variety']                # Dependent variable
```

---

## 4. Split Dataset into Training and Testing Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 5. Train Naïve Bayes Model

```python
model = GaussianNB()

model.fit(X_train, y_train)
```

---

## 6. Predict Output

```python
y_pred = model.predict(X_test)
```

---

## 7. Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

---

## 8. Plot Confusion Matrix

```python
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
```

---

## 9. Calculate Accuracy, Error Rate, Precision and Recall

```python
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred, average='macro')

recall = recall_score(y_test, y_pred, average='macro')

print("Accuracy :", accuracy)

print("Error Rate :", error_rate)

print("Precision :", precision)

print("Recall :", recall)
```

---

## 10. TP, TN, FP, FN

```python
TP = np.diag(cm)

FP = cm.sum(axis=0) - TP

FN = cm.sum(axis=1) - TP

TN = cm.sum() - (FP + FN + TP)

print("True Positive (TP):", TP)

print("False Positive (FP):", FP)

print("False Negative (FN):", FN)

print("True Negative (TN):", TN)
```

---

# Output

* Confusion Matrix displayed
* Accuracy calculated
* Error Rate calculated
* Precision calculated
* Recall calculated
* TP, TN, FP, FN displayed

---

 
