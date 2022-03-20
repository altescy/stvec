import argparse
import dataclasses
import logging
import random
import string
import time
from contextlib import contextmanager
from typing import Iterator, List, Set

from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

from stvec import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TimerResult:
    name: str
    start: float
    end: float = -1

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        return f"<TimerResult: name={self.name} duration={self.duration:.3f}s>"


@contextmanager
def timer(name: str) -> Iterator[TimerResult]:
    try:
        result = TimerResult(name, time.perf_counter())
        yield result
    finally:
        result.end = time.perf_counter()


def build_documents(num_docs: int, vocab_size: int) -> List[str]:
    _vocab: Set[str] = set()
    while len(_vocab) < vocab_size:
        token_length = random.randint(1, 10)
        token = "".join(random.sample(string.ascii_letters, token_length))
        _vocab.add(token)

    vocab = list(_vocab)
    docs: List[str] = []
    for _ in range(num_docs):
        num_tokens = random.randint(10, 500)
        text = " ".join(random.choices(vocab, k=num_tokens))
        docs.append(text)

    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=50000)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.random_seed)

    logger.info("building dataset...")
    docs = build_documents(num_docs=args.num_docs, vocab_size=args.vocab_size)

    logger.info("training with rust tfidf vectorizer...")
    with timer("training rust tfidf") as result:
        vectorizer = TfidfVectorizer()
        vectorizer.fit(docs)
    logger.info("%s", result)

    logger.info("vectorizing with rust tfidf vectorizer...")
    with timer("vectorizing rust tfidf") as result:
        vectorizer.transform(docs)
    logger.info("%s", result)

    logger.info("training with sklearn tfidf vectorizer...")
    with timer("training sklearn tfidf") as result:
        sklearn_vectorizer = SklearnTfidfVectorizer()
        sklearn_vectorizer.fit(docs)
    logger.info("%s", result)

    logger.info("vectorizing with rust tfidf vectorizer...")
    with timer("vectorizing rust tfidf") as result:
        sklearn_vectorizer.transform(docs)
    logger.info("%s", result)

    logger.info("fit_transform with rust tfidf vectorizer...")
    with timer("fit_transform with rust tfidf") as result:
        TfidfVectorizer().fit_transform(docs)
    logger.info("%s", result)

    logger.info("fit_transform with sklearn tfidf vectorizer...")
    with timer("fit_transform with sklearn tfidf") as result:
        SklearnTfidfVectorizer().fit_transform(docs)
    logger.info("%s", result)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    main()
