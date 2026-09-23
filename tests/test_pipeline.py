"""管道层：摄入 / 检索 / RAG 集成验证。"""



def test_ingest_count(components):
    assert components["vectorstore"].count() == 3


def test_retrieval_relevance(components):
    hits = components["retriever"].retrieve("作者是谁", top_k=2)
    assert hits
    assert any("晨星" in h.chunk.text for h in hits)


def test_rag_answer(components):
    res = components["rag"].ask("谁是 aether-ai-core 的作者？")
    assert res.sources
    assert "晨星" in res.answer


def test_rag_empty_query(components):
    res = components["rag"].ask("完全无关的问题 xyz")
    assert isinstance(res.answer, str)


def test_chunking_boundary():
    from aether.pipeline.ingest import chunk_text

    long = "甲" * 1000
    parts = chunk_text(long, size=400, overlap=80)
    assert len(parts) > 1
    assert all(len(p) <= 400 for p in parts)
