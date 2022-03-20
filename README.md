# STVEC

[![Actions Status](https://github.com/altescy/stvec/workflows/CI/badge.svg)](https://github.com/altescy/stvec/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/stvec)](https://github.com/altescy/stvec/blob/master/LICENSE)

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
