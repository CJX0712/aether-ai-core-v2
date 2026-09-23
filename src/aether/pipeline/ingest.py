"""摄入管道：文档加载 -> 切分 -> 嵌入 -> 写入向量库。

单一职责：把原始文档变成可被检索的向量索引。各步骤均通过接口，
因此可独立替换为更高级的加载器 / 切分器 / 嵌入器。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from aether.core.interfaces import EmbeddingModel, Ingestor, VectorStore
from aether.core.logging import get_logger
from aether.core.models import Chunk, Document

logger = get_logger("aether.pipeline.ingest")


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """按字符滑动窗口切分（中文友好，不依赖分词）。"""

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: list[str] = []
    start = 0
    while start < len(text):
        out.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return out


def load_file(path: str) -> list[Document]:
    """按扩展名加载文档。PDF 需 pypdf（可选）。"""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        content = p.read_text(encoding="utf-8", errors="ignore")
        return [Document(id=_sha(p.name + content[:32]), content=content, source=str(p))]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [
                Document(id=_sha(str(i) + json.dumps(d, ensure_ascii=False)[:32]),
                         content=json.dumps(d, ensure_ascii=False), source=str(p))
                for i, d in enumerate(data)
            ]
        return [Document(id=_sha(p.name), content=json.dumps(data, ensure_ascii=False), source=str(p))]
    if suffix == ".pdf":
        try:
            import pypdf
        except Exception as exc:
            raise RuntimeError(f"读取 PDF 需要 pypdf：pip install pypdf（{exc}）") from exc
        reader = pypdf.PdfReader(str(p))
        content = "\n".join((page.extract_text() or "") for page in reader.pages)
        return [Document(id=_sha(p.name + content[:32]), content=content, source=str(p))]
    raise ValueError(f"不支持的文件类型：{suffix}")


class DefaultIngestor(Ingestor):
    def __init__(
        self,
        embedding: EmbeddingModel,
        vectorstore: VectorStore,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ) -> None:
        self._embedding = embedding
        self._vectorstore = vectorstore
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, documents: Sequence[Document]) -> int:
        ids: list[str] = []
        vectors: list[list[float]] = []
        payloads: list[dict] = []
        all_texts: list[str] = []
        chunks: list[Chunk] = []

        for doc in documents:
            for offset, piece in enumerate(chunk_text(doc.content, self._chunk_size, self._chunk_overlap)):
                cid = f"{doc.id}#{offset}"
                chunk = Chunk(id=cid, document_id=doc.id, text=piece, metadata=doc.metadata)
                chunks.append(chunk)
                all_texts.append(piece)

        if not chunks:
            return 0

        self._embedding.fit(all_texts)
        emb_map = self._embedding.embed(all_texts, update=False)

        for chunk, vec in zip(chunks, emb_map):
            ids.append(chunk.id)
            vectors.append(vec)
            payloads.append(
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

        self._vectorstore.add(ids, vectors, payloads)
        logger.info("摄入完成：文档=%d 切片=%d", len(documents), len(chunks))
        return len(chunks)
