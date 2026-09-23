"""检索工具：把知识库检索封装为 Agent 可调用的能力。"""

from __future__ import annotations

from aether.core.interfaces import Retriever, Tool


class RetrieverTool(Tool):
    name = "retriever_tool"
    description = "在知识库中检索与问题相关的文档片段，输入为查询文本。"
    input_param = "query"

    def __init__(self, retriever: Retriever, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def run(self, query: str = "", **kwargs: object) -> str:
        q = query or str(kwargs)
        chunks = self._retriever.retrieve(q, top_k=self._top_k)
        if not chunks:
            return "（知识库中未检索到相关片段）"
        lines = []
        for i, rc in enumerate(chunks, 1):
            src = rc.chunk.metadata.get("source") or rc.chunk.document_id
            lines.append(f"[{i}] 来源:{src} 相似度:{rc.score:.3f}\n{rc.chunk.text}")
        return "\n\n".join(lines)
