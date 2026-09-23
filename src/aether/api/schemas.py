"""API 请求/响应模型。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IngestRequest(BaseModel):
    text: str | None = None
    source: str = "api"
    documents: list[dict[str, Any]] | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class ChatRequest(BaseModel):
    task: str


class IngestResponse(BaseModel):
    chunks: int
    status: str = "ok"


class InfoResponse(BaseModel):
    name: str = "aether-ai-core"
    version: str
    author: str = "晨星"
    embedding: str
    vectorstore: str
    llm: str
