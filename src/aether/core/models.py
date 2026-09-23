"""核心数据模型。

所有跨模块数据交换均通过本模块定义的 Pydantic 模型完成，
保证接口契约稳定、可独立验证。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """对话角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Document(BaseModel):
    """一份待摄入的源文档。"""

    id: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """文档切片。检索的最小单元。"""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """向量库单次命中。"""

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """检索器产出：带分数的切片。"""

    chunk: Chunk
    score: float


class ChatMessage(BaseModel):
    """一轮对话消息。"""

    role: Role
    content: str
    name: str | None = None


class GenerationResult(BaseModel):
    """生成结果，含可追溯来源。"""

    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class ToolCall(BaseModel):
    """Agent 发起的一次工具调用。"""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    """Agent 执行结果。"""

    answer: str
    steps: list[ToolCall] = Field(default_factory=list)
    model: str = ""


class HealthStatus(BaseModel):
    """健康检查快照。"""

    status: str = "ok"
    components: dict[str, str] = Field(default_factory=dict)
    version: str = __import__("aether").__version__
