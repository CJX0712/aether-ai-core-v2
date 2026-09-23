"""检索管道：查询 -> 相关切片。"""

from __future__ import annotations

from aether.core.interfaces import EmbeddingModel, Retriever, VectorStore
from aether.core.logging import get_logger
from aether.core.models import Chunk, RetrievedChunk

logger = get_logger("aether.pipeline.retrieval")


class DefaultRetriever(Retriever):
    def __init__(
        self, embedding: EmbeddingModel, vectorstore: VectorStore, top_k: int = 5
    ) -> None:
        self._embedding = embedding
        self._vectorstore = vectorstore
        self._top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self._top_k
        if k <= 0 or self._vectorstore.count() == 0:
            return []
        qv = self._embedding.embed([query], update=False)[0]
        results = self._vectorstore.search(qv, top_k=k)
        out: list[RetrievedChunk] = []
        for r in results:
            payload = r.payload
            chunk = Chunk(
                id=payload.get("id", r.id),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
            )
            out.append(RetrievedChunk(chunk=chunk, score=r.score))
        logger.info("检索完成：query='%s...' 命中=%d", query[:20], len(out))
        return out
