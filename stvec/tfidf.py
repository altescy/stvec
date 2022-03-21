from typing import Any, Dict, List, Literal, Optional, Set, Union

from scipy import sparse
from sklearn.base import TransformerMixin
from sklearn.preprocessing import normalize

from .stvec import TfidfVectorizer as _TfidfVectorizer


class TfidfVectorizer(TransformerMixin):  # type: ignore[misc]
    def __init__(
        self,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
        use_idf: bool = True,
        norm: Literal["l1", "l2", None] = "l2",
        stop_words: Optional[Set[str]] = None,
    ) -> None:
        super().__init__()
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.norm = norm
        self.stop_words = stop_words or set()
        self._vectorizer: Optional[_TfidfVectorizer] = None

    def _get_vectorizer(self) -> _TfidfVectorizer:
        if self._vectorizer is None:
            raise RuntimeError("TfidfVectorizer have not been trained yet!")
        return self._vectorizer

    def get_output_dim(self) -> int:
        vectorizer = self._get_vectorizer()
        return int(vectorizer.get_output_dim())

    def fit(self, docs: List[str]) -> "TfidfVectorizer":
        num_docs = len(docs)
        min_df = int(self.min_df if isinstance(self.min_df, int) else num_docs * self.min_df)
        max_df = int(self.max_df if isinstance(self.max_df, int) else num_docs * self.max_df)
        self._vectorizer = _TfidfVectorizer(min_df, max_df, self.use_idf, self.stop_words)
        self._vectorizer.train(docs)
        return self

    def transform(self, docs: List[str]) -> sparse.csr_matrix:
        vectorizer = self._get_vectorizer()
        row, col, data = vectorizer.vectorize(docs)
        num_samples = len(docs)
        num_features = self.get_output_dim()
        output = sparse.csr_matrix((data, (row, col)), shape=(num_samples, num_features))
        if self.norm:
            output = normalize(output, self.norm)
        return output

    def __getstate__(self) -> Dict[str, Any]:
        state = {
            "min_df": self.min_df,
            "max_df": self.max_df,
            "use_idf": self.use_idf,
            "norm": self.norm,
        }
        if self._vectorizer is not None:
            state["vectorizer"] = self._vectorizer.to_params()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.min_df = state["min_df"]
        self.max_df = state["max_df"]
        self.use_idf = state["use_idf"]
        self.norm = state["norm"]
        if "vectorizer" in state:
            self._vectorizer = _TfidfVectorizer.from_params(state["vectorizer"])
