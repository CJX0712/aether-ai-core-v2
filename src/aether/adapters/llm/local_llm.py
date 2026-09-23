"""本地推理适配器：ctransformers（GGUF 量化模型）。

在本地硬件上运行开源大模型，数据不出域。依赖可选。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aether.core.interfaces import LLMBackend


class LocalLLM(LLMBackend):
    def __init__(self, model_path: str, max_new_tokens: int = 512) -> None:
        try:
            from ctransformers import AutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(f"未安装 ctransformers：pip install ctransformers（{exc}）") from exc
        self._llm = AutoModelForCausalLM.from_pretrained(model_path, model_type="llama")
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens

    @property
    def name(self) -> str:
        return self._model_path

    def complete(self, messages: Sequence[Any], **kwargs: Any) -> str:
        prompt = "\n".join(
            f"{getattr(m, 'role', m.get('role'))}: {getattr(m, 'content', m.get('content'))}"
            for m in messages
        )
        return self._llm(prompt, max_new_tokens=kwargs.get("max_tokens", self._max_new_tokens))
