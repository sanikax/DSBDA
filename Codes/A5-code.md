# A5 - Data Analytics-II

---

# Problem Statement

Implement Logistic Regression using Python to perform classification.

Compute:

* TP
* FP
* TN
* FN
* Accuracy
* Error Rate
* Precision
* Recall

---

# Pre-requisites

```bash id="xjv8q3"
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# Code

## 1. Import Libraries

```python id="nm2j8o"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
```

---

## 2. Load Inbuilt Dataset

```python id="sj0o4d"
data = load_breast_cancer()

# Convert into DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Add target column
df['target'] = data.target

print(df.head())
```

---

## 3. Define Features and Target

```python id="g53cn4"
X = df.drop('target', axis=1)

y = df['target']
```

---

## 4. Split the Dataset

```python id="efc2pv"
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42
)
```

---

## 5. Feature Scaling

```python id="btxo8z"
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
```

---

## 6. Train Logistic Regression Model

```python id="q3jv2x"
model = LogisticRegression()

model.fit(X_train, y_train)
```

---

## 7. Make Predictions

```python id="z4a8lc"
y_pred = model.predict(X_test)
```

---

## 8. Confusion Matrix

```python id="7h5l8k"
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
```

---

## 9. Find TP, FP, TN, FN

```python id="q1r6po"
TN, FP, FN, TP = cm.ravel()

print("True Positive =", TP)
print("False Positive =", FP)
print("True Negative =", TN)
print("False Negative =", FN)
```

---

## 10. Calculate Accuracy, Error Rate, Precision, Recall

```python id="l3y7dc"
accuracy = accuracy_score(y_test, y_pred)

error_rate = 1 - accuracy

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

print("Accuracy =", accuracy)

print("Error Rate =", error_rate)

print("Precision =", precision)

print("Recall =", recall)
```

---

## 11. Visualize Confusion Matrix

```python id="mw6p3e"
sns.heatmap(cm, annot=True, fmt='d')

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()
```

---

# ✅ Output

* Logistic Regression model created
* Predictions performed
* Confusion matrix generated
* Accuracy, Precision, Recall calculated
* Heatmap visualized

---

 