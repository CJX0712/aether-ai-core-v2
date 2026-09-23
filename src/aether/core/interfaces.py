"""模块接口契约（单一职责 + 可独立验证）。

任何具体实现都须继承以下抽象基类。接口层不依赖任何具体后端，
因此每个模块可在无其余模块在场时被单独单元测试。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aether.core.models import (
    Document,
    RetrievedChunk,
    SearchResult,
)


class EmbeddingModel(ABC):
    """文本向量化。单一职责：文本 -> 向量。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量将文本编码为向量。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class VectorStore(ABC):
    """向量索引与近邻检索。单一职责：存向量 + 检索。"""

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        """写入向量与附属载荷。"""

    @abstractmethod
    def search(
        self, query_vector: Sequence[float], top_k: int = 5
    ) -> list[SearchResult]:
        """返回 top_k 个最相似命中（分数越高越相关）。"""

    @abstractmethod
    def count(self) -> int:
        """当前索引条目数。"""

    @abstractmethod
    def clear(self) -> None:
        """清空索引。"""


class LLMBackend(ABC):
    """语言模型推理。单一职责：消息序列 -> 文本补全。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型标识。"""

    @abstractmethod
    def complete(self, messages: Sequence[Any], **kwargs: Any) -> str:
        """给定对话消息，返回模型生成的文本。"""

    def complete_text(self, prompt: str, **kwargs: Any) -> str:
        from aether.core.models import ChatMessage, Role

        return self.complete(
            [ChatMessage(role=Role.SYSTEM, content=prompt)], **kwargs
        )


class Retriever(ABC):
    """检索编排。单一职责：查询 -> 相关切片。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """返回与查询相关的切片（按相关性降序）。"""


class Ingestor(ABC):
    """文档摄入。单一职责：文档 -> 向量库（含切分与嵌入）。"""

    @abstractmethod
    def ingest(self, documents: Sequence[Document]) -> int:
        """摄入文档，返回写入的切片数。"""


class Tool(ABC):
    """Agent 可调用的工具。单一职责：命名能力 + 执行。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具，返回文本结果。"""

    def spec(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


class Agent(ABC):
    """任务推理与执行。单一职责：任务 -> 带工具调用的结果。"""

    @abstractmethod
    def run(self, task: str) -> Any:
        """执行任务，返回 AgentResult。"""
