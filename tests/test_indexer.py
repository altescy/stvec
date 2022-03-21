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
