import pickle
from pathlib import Path

from stvec import Indexer


def test_indexer() -> None:
    docs = [
        "there ain't no such thing as a free lunch",
        "there are some apples on the table",
    ]

    indexer = Indexer()
    indexer.train(docs)

    inputs = [
        "there ain't no such thing as a free lunch",
        "there are some lemons on the table",
    ]
    indices, mask = indexer.vectorize(inputs)

    assert indices.shape == (2, 9)
    assert mask.sum(1).tolist() == [9, 7]
    assert (indices == indexer.unk()).sum() == 1


def test_pickle_indexer(tmp_path: Path) -> None:
    docs = [
        "there ain't no such thing as a free lunch",
        "there are some apples on the table",
    ]

    indexer = Indexer()
    indexer.train(docs)

    with open(tmp_path / "indexer.pkl", "wb") as pklfile:
        pickle.dump(indexer, pklfile)

    with open(tmp_path / "indexer.pkl", "rb") as pklfile:
        deserialized = pickle.load(pklfile)

    inputs = [
        "there ain't no such thing as a free lunch",
        "there are some lemons on the table",
    ]
    indices, mask = deserialized.vectorize(inputs)

    assert indices.shape == (2, 9)
