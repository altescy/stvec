from typing import List, Optional, Tuple, Union, cast

import numpy

from .stvec import Indexer as _Indexer


class Indexer:
    def __init__(
        self,
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0,
    ) -> None:
        self.min_df = min_df
        self.max_df = max_df
        self._indexer: Optional[_Indexer] = None

    def _get_indexer(self) -> _Indexer:
        if self._indexer is None:
            raise RuntimeError("Indexer have not been trained yet!")
        return self._indexer

    def bos(self) -> int:
        return int(self._get_indexer().bos())

    def eos(self) -> int:
        return int(self._get_indexer().eos())

    def pad(self) -> int:
        return int(self._get_indexer().pad())

    def unk(self) -> int:
        return int(self._get_indexer().unk())

    def train(self, docs: List[str]) -> None:
        num_docs = len(docs)
        min_df = int(self.min_df if isinstance(self.min_df, int) else self.min_df * num_docs)
        max_df = int(self.max_df if isinstance(self.max_df, int) else self.max_df * num_docs)
        self._indexer = _Indexer(min_df, max_df)
        self._indexer.train(docs)

    def vectorize(self, docs: List[str]) -> Tuple[numpy.ndarray, numpy.ndarray]:
        output, mask = self._get_indexer().vectorize(docs)
        return cast(numpy.ndarray, output), cast(numpy.ndarray, mask)
