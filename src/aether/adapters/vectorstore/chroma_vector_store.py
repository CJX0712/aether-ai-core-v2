"""世界顶级向量库适配器：Chroma（持久化、可扩展）。

复用业界领先开源向量数据库。依赖可选，缺失时容器给出明确提示。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aether.core.interfaces import VectorStore
from aether.core.models import SearchResult


class ChromaVectorStore(VectorStore):
    def __init__(self, name: str = "aether", path: str = ".chroma") -> None:
        try:
            import chromadb
        except Exception as exc:
            raise RuntimeError(f"未安装 chromadb：pip install chromadb（{exc}）") from exc
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=name)

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        documents = [p.get("text", "") for p in payloads]
        metadatas = [{k: v for k, v in p.get("metadata", {}).items()} for p in payloads]
        self._collection.add(
            ids=list(ids),
            embeddings=[list(v) for v in vectors],
            documents=documents,
            metadatas=metadatas,
        )

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        res = self._collection.query(
            query_embeddings=[list(query_vector)], n_results=max(1, top_k)
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        documents = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        out: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            dist = distances[i] if i < len(distances) else 0.0
            score = 1.0 / (1.0 + max(0.0, float(dist)))
            meta = metadatas[i] if i < len(metadatas) else {}
            text = documents[i] if i < len(documents) else ""
            out.append(
                SearchResult(
                    id=doc_id, score=score, payload={"text": text, "metadata": meta}
                )
            )
        return out

    def count(self) -> int:
        return int(self._collection.count())

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)
