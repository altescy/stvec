from importlib.metadata import version

from stvec.indexer import Indexer  # noqa: F401
from stvec.tfidf import TfidfVectorizer  # noqa: F401

__version__ = version("stvec")
