"""core：接口契约、数据模型、配置与容器。"""

from aether.core.interfaces import (
    Agent,
    EmbeddingModel,
    Ingestor,
    LLMBackend,
    Retriever,
    Tool,
    VectorStore,
)
from aether.core.models import (
    AgentResult,
    Chunk,
    Document,
    GenerationResult,
    RetrievedChunk,
    SearchResult,
)

__all__ = [
    "Agent",
    "AgentResult",
    "Chunk",
    "Document",
    "EmbeddingModel",
    "GenerationResult",
    "Ingestor",
    "LLMBackend",
    "RetrievedChunk",
    "Retriever",
    "SearchResult",
    "Tool",
    "VectorStore",
]
