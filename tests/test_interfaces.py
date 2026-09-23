"""接口契约与默认适配器单元验证。"""

from aether.adapters.embeddings.tfidf_embedding import TfidfEmbedding
from aether.adapters.vectorstore.memory_vector_store import MemoryVectorStore


def test_embedding_shape_and_norm():
    emb = TfidfEmbedding(dim=64)
    emb.fit(["hello world", "world of python"])
    vec = emb.embed(["hello world"], update=False)[0]
    assert len(vec) == 64
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-6  # L2 归一化


def test_embedding_similarity():
    emb = TfidfEmbedding(dim=128)
    emb.fit(["机器学习 模型 训练", "深度学习 神经网络", "完全不相关的句子 苹果 香蕉"])
    a = emb.embed(["机器学习 模型 训练"], update=False)[0]
    b = emb.embed(["深度学习 神经网络"], update=False)[0]
    c = emb.embed(["苹果 香蕉 水果"], update=False)[0]
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))
    assert dot(a, b) > dot(a, c)


def test_vectorstore_add_search_clear():
    vs = MemoryVectorStore()
    assert vs.count() == 0
    vs.add(["1", "2"], [[1.0, 0.0], [0.0, 1.0]], [{"text": "a"}, {"text": "b"}])
    assert vs.count() == 2
    res = vs.search([1.0, 0.0], top_k=1)
    assert res[0].id == "1"
    assert res[0].payload["text"] == "a"
    vs.clear()
    assert vs.count() == 0


def test_vectorstore_empty_search():
    assert MemoryVectorStore().search([1.0, 0.0]) == []
