"""世界顶级推理适配器：OpenAI 兼容 API。

一行切换底层引擎：本地 vLLM / Ollama，或云端 OpenAI。
复用业界最成熟的兼容协议，零自研推理代码。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from aether.core.interfaces import LLMBackend
from aether.core.logging import get_logger
from aether.core.models import Role

logger = get_logger("aether.llm.openai")


def _to_dict(m: Any) -> dict[str, Any]:
    if isinstance(m, dict):
        role = m.get("role")
        content = m.get("content")
    else:
        role = getattr(m, "role", "")
        content = getattr(m, "content", "")
    if isinstance(role, Role):
        role = role.value
    return {"role": str(role), "content": str(content)}


class OpenAICompatibleLLM(LLMBackend):
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def name(self) -> str:
        return self._model

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> str:
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        payload = {
            "model": self._model,
            "messages": [_to_dict(m) for m in messages],
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
        }
        url = f"{self._base}/chat/completions"
        with httpx.Client(timeout=self._timeout, headers=headers) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]
