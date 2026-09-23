"""默认可运行 Embedding：哈希 TF-IDF 向量器。

零重依赖、确定性、可复现。将文本映射为固定维度 L2 归一化向量，
语义相近的文本因共享词项而余弦相似度高。世界顶级语义向量请选用
SentenceTransformerEmbedding 适配器。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence

from aether.core.interfaces import EmbeddingModel
from aether.core.logging import get_logger

logger = get_logger("aether.embed.tfidf")

_TOKEN_RE = re.compile(r"[a-z0-9]+|[一-龥]")


class TfidfEmbedding(EmbeddingModel):
    def __init__(self, dim: int = 256) -> None:
        self._dim = dim
        self._df: dict[str, int] = {}
        self._n_docs = 0

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def _bucket(self, term: str) -> int:
        digest = hashlib.md5(term.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % self._dim

    def fit(self, corpus: Iterable[str]) -> None:
        """在摄入语料上学习 IDF（先行于 embed）。"""

        for doc in corpus:
            for term in set(self._tokens(doc)):
                self._df[term] = self._df.get(term, 0) + 1
            self._n_docs += 1
        logger.info("TF-IDF 拟合完成：词表=%d 文档=%d", len(self._df), self._n_docs)

    def embed(self, texts: Sequence[str], update: bool = True) -> list[list[float]]:
        if update:
            self.fit(texts)
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            tf: dict[str, int] = {}
            for term in self._tokens(text):
                tf[term] = tf.get(term, 0) + 1
            norm_sq = 0.0
            for term, count in tf.items():
                df = self._df.get(term, 0)
                idf = math.log((1 + self._n_docs) / (1 + df)) + 1.0
                weight = (1.0 + math.log(count)) * idf
                vec[self._bucket(term)] += weight
                norm_sq += weight * weight
            if norm_sq > 0.0:
                scale = 1.0 / math.sqrt(norm_sq)
                vec = [v * scale for v in vec]
            out.append(vec)
        return out
