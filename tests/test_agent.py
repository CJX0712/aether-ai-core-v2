"""Agent 层：ReAct 工具调用验证（默认 mock 栈驱动）。"""


def test_agent_calculator(components):
    res = components["agent"].run("请帮我计算 (128+64)*3")
    assert res.steps, "Agent 未发起工具调用"
    assert any(t.tool == "calculator" for t in res.steps)
    assert "576" in res.answer or "Final Answer" in res.answer


def test_agent_retriever(components):
    res = components["agent"].run("aether-ai-core 是谁开发的？")
    assert any(t.tool == "retriever_tool" for t in res.steps)
    assert "晨星" in res.answer
