"""世界顶级向量库适配器：FAISS（Facebook AI Similarity Search）。

工业级近邻检索，支持大规模向量。依赖可选，缺失时容器给出明确提示。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aether.core.interfaces import VectorStore
from aether.core.models import SearchResult


class FaissVectorStore(VectorStore):
    def __init__(self, dim: int = 256) -> None:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError(f"未安装 faiss-cpu：pip install faiss-cpu（{exc}）") from exc
        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._payloads: list[dict[str, Any]] = []

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        import numpy as np

        mat = np.asarray(list(vectors), dtype="float32")
        if mat.shape[1] != self._dim:
            raise ValueError(f"维度不符：期望 {self._dim}，收到 {mat.shape[1]}")
        self._index.add(mat)
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        import numpy as np

        if self._index.ntotal == 0:
            return []
        q = np.asarray([list(query_vector)], dtype="float32")
        k = min(top_k, self._index.ntotal)
        scores, idx = self._index.search(q, k)
        out: list[SearchResult] = []
        for rank, i in enumerate(idx[0]):
            if i < 0:
                continue
            out.append(
                SearchResult(
                    id=self._ids[i],
                    score=float(scores[0][rank]),
                    payload=self._payloads[i],
                )
            )
        return out

    def count(self) -> int:
        return int(self._index.ntotal)

    def clear(self) -> None:
        self._index.reset()
        self._ids = []
        self._payloads = []
