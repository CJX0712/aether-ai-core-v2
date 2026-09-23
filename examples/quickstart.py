"""aether-ai-core 快速上手示例（默认栈，无需 GPU）。

运行：python examples/quickstart.py
作者：晨星
"""

from aether.core.config import load_config
from aether.core.container import build_container
from aether.core.models import Document


def main() -> None:
    comp = build_container(load_config())

    corpus = [
        Document(id="d1", content="晨星是 aether-ai-core 项目的作者，负责整体架构与文档。", source="intro"),
        Document(id="d2", content="该系统采用接口驱动的模块化设计，embedding、向量库、大模型均可插拔替换。", source="design"),
        Document(id="d3", content="默认栈无需 GPU 即可运行，世界顶级后端以可选适配器形式接入。", source="deploy"),
    ]
    n = comp["ingestor"].ingest(corpus)
    print(f"[摄入] 切片数 = {n}")

    q = comp["rag"].ask("谁是 aether-ai-core 的作者？")
    print(f"[RAG] 答案：{q.answer}")
    print(f"[RAG] 来源数：{len(q.sources)}")

    a = comp["agent"].run("aether-ai-core 的作者还负责什么？并计算 (100+20)*2")
    print(f"[Agent] 回答：{a.answer}")
    print(f"[Agent] 推理步：{[s.tool for s in a.steps]}")


if __name__ == "__main__":
    main()
