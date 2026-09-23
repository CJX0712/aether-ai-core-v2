"""端到端集成：摄入 -> 检索 -> RAG -> Agent 全链路跑通。"""


def test_end_to_end(components):
    # 1) 摄入
    assert components["vectorstore"].count() == 3
    # 2) 检索
    hits = components["retriever"].retrieve("模块化 设计", top_k=3)
    assert hits
    # 3) RAG 生成
    rag = components["rag"].ask("系统采用什么设计？")
    assert "接口" in rag.answer or rag.sources
    # 4) Agent 多步推理
    agent = components["agent"].run("aether-ai-core 的作者还负责什么？")
    assert agent.steps
    assert "晨星" in agent.answer
