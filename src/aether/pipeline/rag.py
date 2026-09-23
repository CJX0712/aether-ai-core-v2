"""生成管道：检索上下文 + 大模型 -> 可追溯答案（RAG）。"""

from __future__ import annotations

from aether.core.interfaces import LLMBackend, Retriever
from aether.core.logging import get_logger
from aether.core.models import ChatMessage, GenerationResult, Role

logger = get_logger("aether.pipeline.rag")

_SYSTEM_TEMPLATE = (
    "你是一个严谨的问答助手。仅使用下方提供的上下文回答问题，"
    "若上下文不足请如实说明。请使用与问题相同的语言作答。\n\n"
    "上下文：\n{context}"
)


class RAGPipeline:
    def __init__(self, retriever: Retriever, llm: LLMBackend, model: str = "") -> None:
        self._retriever = retriever
        self._llm = llm
        self._model = model or llm.name

    def _build_context(self, chunks) -> str:
        lines = []
        for i, rc in enumerate(chunks, 1):
            src = rc.chunk.metadata.get("source") or rc.chunk.document_id
            lines.append(f"[{i}] (来源:{src}) {rc.chunk.text}")
        return "\n".join(lines) if lines else "（无相关上下文）"

    def ask(self, query: str, top_k: int | None = None) -> GenerationResult:
        sources = self._retriever.retrieve(query, top_k=top_k)
        context = self._build_context(sources)
        system = ChatMessage(role=Role.SYSTEM, content=_SYSTEM_TEMPLATE.format(context=context))
        user = ChatMessage(role=Role.USER, content=query)
        answer = self._llm.complete([system, user])
        prompt_tokens = len(context.split()) + len(query.split())
        completion_tokens = len(answer.split())
        return GenerationResult(
            answer=answer,
            sources=sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=self._model,
        )
