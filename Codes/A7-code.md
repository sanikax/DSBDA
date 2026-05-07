# A7 - Text Analytics

---

# Problem Statement

Perform the following operations using Python:

1. Tokenization
2. POS Tagging
3. Stop words removal
4. Stemming
5. Lemmatization
6. Calculate TF-IDF (Term Frequency & Inverse Document Frequency)

---

# Pre-requisites

Install required libraries:

```bash
pip install nltk scikit-learn
```

---

# 1. Import Libraries

```python
import nltk
import re

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
```

---

# 2. Download Required Resources

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

---

# 3. Sample Text Document

```python
text = """
Hello everyone. This is a simple text analytics practical.
I love Python programming and machine learning.
Python makes data analysis easy and interesting.
"""
```

---

# 4. Sentence Tokenization

```python
sentences = sent_tokenize(text)

print("Sentence Tokenization:")
print(sentences)
```

---

# 5. Word Tokenization

```python
words = word_tokenize(text)

print("Word Tokenization:")
print(words)
```

---

# 6. Remove Punctuation

```python
clean_text = re.sub(r'[^a-zA-Z ]', '', text)

print("Text after removing punctuation:")
print(clean_text)
```

---

# 7. Remove Stop Words

```python
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(clean_text.lower())

filtered_words = []

for word in tokens:
    if word not in stop_words:
        filtered_words.append(word)

print("Words after removing stop words:")
print(filtered_words)
```

---

# 8. Stemming

```python
ps = PorterStemmer()

print("Stemming:")

for word in filtered_words:
    print(word, "->", ps.stem(word))
```

---

# 9. Lemmatization

```python
lemmatizer = WordNetLemmatizer()

print("Lemmatization:")

for word in filtered_words:
    print(word, "->", lemmatizer.lemmatize(word))
```

---

# 10. POS Tagging

```python
pos = nltk.pos_tag(word_tokenize(clean_text))

print("POS Tagging:")
print(pos)
```

---

# 11. TF-IDF Calculation

```python
document = [clean_text]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(document)

feature_names = vectorizer.get_feature_names_out()

print("TF-IDF Values:")
print(feature_names)

print(tfidf_matrix.toarray())
```

---

# ✅ Output

The program successfully performs:

* Tokenization
* POS Tagging
* Stop words removal
* Stemming
* Lemmatization
* TF-IDF calculation

---
 
