"""默认可运行向量库：内存余弦索引（基于 numpy）。

零外部依赖、确定性、可复现。向量已 L2 归一化时余弦=点积。
世界顶级规模请选用 ChromaVectorStore / FaissVectorStore 适配器。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aether.core.interfaces import VectorStore
from aether.core.models import SearchResult


class MemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._payloads: list[dict[str, Any]] = []

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        mat = np.asarray(list(vectors), dtype=np.float64)
        if self._vectors is None:
            self._vectors = mat
        else:
            self._vectors = np.vstack([self._vectors, mat])
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.asarray(query_vector, dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self._vectors @ q
        k = min(top_k, len(self._ids))
        idx = np.argpartition(-scores, range(k))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [
            SearchResult(id=self._ids[i], score=float(scores[i]), payload=self._payloads[i])
            for i in idx
        ]

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids = []
        self._vectors = None
        self._payloads = []
