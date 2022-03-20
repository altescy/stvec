# STVEC

Simple Text Vectorizer written in Rust

```python
from stvec import TfidfVectorizer

docs = [
    "this is a first sentence",
    "this is a second sentence",
]
vectorizer = TfidfVectorizer()
output = vectorizer.fit_transform(docs)
```
