import pickle
from pathlib import Path

from stvec import TfidfVectorizer


def test_tfidf_vectorizer() -> None:
    texts = [
        "this is a first sentence",
        "this is a second sentence",
    ]
    vectorizer = TfidfVectorizer()
    output = vectorizer.fit_transform(texts)

    assert output.shape == (2, 6)


def test_pickle_tfidf_vectorizer(tmp_path: Path) -> None:
    texts = [
        "this is a first sentence",
        "this is a second sentence",
    ]
    vectorizer = TfidfVectorizer().fit(texts)

    with open(tmp_path / "tfidf.pkl", "wb") as pklfile:
        pickle.dump(vectorizer, pklfile)

    with open(tmp_path / "tfidf.pkl", "rb") as pklfile:
        deserialized = pickle.load(pklfile)

    assert isinstance(deserialized, TfidfVectorizer)

    output = deserialized.transform(texts)
    assert output.shape == (2, 6)
