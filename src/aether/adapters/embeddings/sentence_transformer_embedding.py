"""世界顶级语义向量适配器：Sentence-Transformers。

复用业界领先开源模型（如 BAAI/bge、sentence-transformers/all-MiniLM），
输出高保真句向量。依赖可选，缺失时容器会给出明确安装提示。
"""

from __future__ import annotations

from collections.abc import Sequence

from aether.core.interfaces import EmbeddingModel


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                f"未安装 sentence-transformers：pip install sentence-transformers（{exc}）"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._name = model_name

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str], update: bool = True) -> list[list[float]]:
        # 语义模型无需在线拟合；update 参数仅为接口兼容保留。
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vectors]
