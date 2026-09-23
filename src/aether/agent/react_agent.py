"""ReAct Agent：推理-行动循环 + 工具调用。

单一职责：把任务分解为多步「思考-行动-观察」，直到产出 Final Answer。
默认栈（MockLLM）亦可驱动：MockLLM 会首轮选工具、工具结果回传后给 Final Answer。
接入真实大模型（OpenAICompatibleLLM）后即为完整的 ReAct 推理。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from aether.core.interfaces import Agent, LLMBackend, Tool
from aether.core.logging import get_logger
from aether.core.models import AgentResult, ChatMessage, Role, ToolCall

logger = get_logger("aether.agent.react")

_SYSTEM_PROMPT = (
    "你是一个使用工具解决问题的智能体。每步按如下格式输出：\n"
    "Thought: 你的思考\n"
    "Action: 工具名\n"
    "Action Input: 工具输入\n"
    "或最终输出：\n"
    "Final Answer: 最终答案\n"
    "可用工具：\n{tools}\n"
    "注意：Action 必须是上述工具名之一。"
)


class ReActAgent(Agent):
    def __init__(self, llm: LLMBackend, tools: Sequence[Tool], model: str = "", max_steps: int = 5) -> None:
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._model = model or llm.name
        self._max_steps = max_steps

    def _tool_spec(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    @staticmethod
    def _parse_final(text: str) -> str | None:
        m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_action(text: str):
        m = re.search(r"Action:\s*(\w+)", text)
        if not m:
            return None
        name = m.group(1)
        mi = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
        inp = mi.group(1).strip() if mi else ""
        return name, inp

    def run(self, task: str) -> AgentResult:
        system = ChatMessage(role=Role.SYSTEM, content=_SYSTEM_PROMPT.format(tools=self._tool_spec()))
        messages: list[ChatMessage] = [system, ChatMessage(role=Role.USER, content=task)]
        steps: list[ToolCall] = []

        for step in range(self._max_steps):
            raw = self._llm.complete(messages)
            messages.append(ChatMessage(role=Role.ASSISTANT, content=raw))

            final = self._parse_final(raw)
            if final:
                logger.info("Agent 在第 %d 步给出 Final Answer", step + 1)
                return AgentResult(answer=final, steps=steps, model=self._model)

            parsed = self._parse_action(raw)
            if not parsed:
                logger.warning("Agent 第 %d 步未解析出 Action/Final Answer", step + 1)
                return AgentResult(answer=raw, steps=steps, model=self._model)

            name, inp = parsed
            tool = self._tools.get(name)
            if tool is None:
                result = f"未知工具：{name}"
                error = result
            else:
                try:
                    result = tool.run(**{tool.input_param: inp})
                    error = None
                except Exception as exc:  # noqa: BLE001
                    result = f"工具执行错误：{exc}"
                    error = result
            steps.append(ToolCall(tool=name, args={tool.input_param: inp} if tool else {}, result=result, error=error))
            messages.append(ChatMessage(role=Role.TOOL, content=result, name=name))

        return AgentResult(
            answer="已达到最大步数仍未产出 Final Answer。",
            steps=steps,
            model=self._model,
        )
