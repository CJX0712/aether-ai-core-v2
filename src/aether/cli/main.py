"""命令行入口：ingest / query / chat / serve / health / eval。

示例：
  python -m aether eval
  python -m aether ingest --text "晨星是本项目作者"
  python -m aether query --query "谁是作者？"
  python -m aether chat --task "计算 (128+64)*3"
  python -m aether serve
"""

from __future__ import annotations

import argparse
import sys

from aether.core.config import load_config
from aether.core.container import build_container
from aether.core.logging import get_logger
from aether.core.models import Document
from aether.pipeline.ingest import load_file

logger = get_logger("aether.cli")


def _build():
    return build_container(load_config())


def cmd_ingest(args) -> int:
    comp = _build()
    docs: list[Document] = []
    if args.file:
        docs.extend(load_file(args.file))
    if args.text:
        docs.append(Document(id="cli", content=args.text, source="cli"))
    if not docs:
        print("请提供 --text 或 --file")
        return 2
    n = comp["ingestor"].ingest(docs)
    print(f"已摄入 {len(docs)} 篇文档，{n} 个切片。")
    return 0


def cmd_query(args) -> int:
    comp = _build()
    res = comp["rag"].ask(args.query, top_k=args.top_k)
    print("答案：")
    print(res.answer)
    print(f"\n来源数：{len(res.sources)} | 模型：{res.model}")
    return 0


def cmd_chat(args) -> int:
    comp = _build()
    res = comp["agent"].run(args.task)
    print("回答：")
    print(res.answer)
    print(f"\n推理步数：{len(res.steps)} | 模型：{res.model}")
    return 0


def cmd_serve(args) -> int:
    cfg = load_config()
    import uvicorn

    uvicorn.run("aether.api.app:app", host=cfg.server_host, port=cfg.server_port, reload=False)
    return 0


def cmd_health(args) -> int:
    cfg = load_config()
    print(
        "组件状态：\n"
        f"  embedding   : {cfg.embedding_backend}\n"
        f"  vectorstore : {cfg.vectorstore_backend}\n"
        f"  llm         : {cfg.llm_backend}\n"
        f"  server      : {cfg.server_host}:{cfg.server_port}"
    )
    return 0


def cmd_eval(args) -> int:
    """一键自检：默认栈端到端跑通。"""
    comp = _build()
    corpus = [
        Document(id="d1", content="晨星是 aether-ai-core 项目的作者，负责整体架构与文档。", source="intro"),
        Document(id="d2", content="该系统采用接口驱动的模块化设计，embedding、向量库、大模型均可插拔替换。", source="design"),
        Document(id="d3", content="默认栈无需 GPU 即可运行，世界顶级后端以可选适配器形式接入。", source="deploy"),
    ]
    n = comp["ingestor"].ingest(corpus)
    assert n == 3, f"摄入切片数异常：{n}"

    q = comp["rag"].ask("谁是 aether-ai-core 的作者？")
    assert q.sources and "晨星" in q.answer, f"RAG 答案未命中作者：{q.answer}"

    a = comp["agent"].run("计算 (128+64)*3")
    assert a.steps, "Agent 未产生工具调用"

    print(f"自检通过 [OK]：摄入={n}, RAG命中作者={'晨星' in q.answer}, Agent步数={len(a.steps)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aether", description="aether-ai-core 命令行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="摄入文档")
    p_ing.add_argument("--text", help="直接文本")
    p_ing.add_argument("--file", help="文件路径(txt/md/json/pdf)")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="RAG 问答")
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--top_k", type=int, default=None)
    p_q.set_defaults(func=cmd_query)

    p_c = sub.add_parser("chat", help="Agent 对话")
    p_c.add_argument("--task", required=True)
    p_c.set_defaults(func=cmd_chat)

    sub.add_parser("serve", help="启动 API 服务").set_defaults(func=cmd_serve)
    sub.add_parser("health", help="查看组件状态").set_defaults(func=cmd_health)
    sub.add_parser("eval", help="一键自检").set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
