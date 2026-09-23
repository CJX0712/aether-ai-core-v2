"""默认可运行 LLM：确定性抽取式 Mock。

无任何外部依赖、可复现，能在无 GPU / 无网络的干净环境中驱动整条链路，
并产出「基于检索上下文」的可追溯答案。接入真实大模型请选用
OpenAICompatibleLLM（兼容 vLLM / Ollama / OpenAI）或 LocalLLM。

对 Agent 友好：能识别 ReAct 提示，首轮选择工具（检索/计算），
工具结果回传后产出 Final Answer，从而让 ReAct 循环在默认栈下也能跑通。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from aether.core.interfaces import LLMBackend
from aether.core.models import Role

_ARITH_RE = re.compile(r"\d+\s*[\+\-\*/]\s*\d+")


def _role(m: Any) -> str:
    if isinstance(m, dict):
        r = m.get("role")
    else:
        r = getattr(m, "role", None)
    if isinstance(r, Role):
        return r.value
    return str(r)


def _content(m: Any) -> str:
    if isinstance(m, dict):
        return str(m.get("content", ""))
    return str(getattr(m, "content", ""))


def _last_user(messages: Sequence[Any]) -> str:
    for m in reversed(list(messages)):
        if _role(m) == Role.USER.value:
            return _content(m)
    return _content(messages[-1]) if messages else ""


class MockLLM(LLMBackend):
    is_mock = True

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return self._model

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[。.!?！？])\s*", text) if s.strip()]

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+|[一-龥]", text.lower())}

    def _extract(self, query: str, context: str) -> str:
        qt = self._tokens(query)
        scored = []
        for sent in self._sentences(context):
            overlap = len(qt & self._tokens(sent))
            if overlap:
                scored.append((overlap, sent))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return context[:200]
        return " ".join(s for _, s in scored[:3])

    @staticmethod
    def _context_of(messages: Sequence[Any]) -> str:
        for m in messages:
            c = _content(m)
            idx = c.find("上下文：")
            if idx >= 0:
                return c[idx + len("上下文："):].strip()
        return ""

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> str:
        full = "\n".join(_content(m) for m in messages)
        is_react = ("可用工具" in full) or ("Action Input:" in full)

        if not is_react:
            # 直答模式（RAG 场景）：从上下文抽取答案。
            query = _last_user(messages)
            ctx = self._context_of(messages)
            if ctx:
                return self._extract(query, ctx)
            return f"[MOCK] 未检索到相关上下文，无法回答：{query}"

        # ReAct 模式：受 Agent 编排驱动。
        has_tool = any(_role(m) == Role.TOOL.value for m in messages)
        query = _last_user(messages)
        if has_tool:
            ctx = "\n".join(
                _content(m) for m in messages if _role(m) == Role.TOOL.value
            )
            return f"Final Answer: {self._extract(query, ctx)}"
        if _ARITH_RE.search(query):
            return (
                f"Thought: 这是一个计算任务，应调用计算器。\n"
                f"Action: calculator\nAction Input: {query}"
            )
        return (
            f"Thought: 应先检索知识库获取相关上下文。\n"
            f"Action: retriever_tool\nAction Input: {query}"
        )
