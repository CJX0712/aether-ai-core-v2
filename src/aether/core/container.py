"""依赖注入容器：按配置装配全部模块。

单一职责：组件创建与接线。运行时才惰性导入具体适配器，
因此缺失可选依赖不会阻断基础栈导入。
"""

from __future__ import annotations

from typing import Any

from aether.core.config import Config
from aether.core.interfaces import (
    Agent,
    EmbeddingModel,
    Ingestor,
    LLMBackend,
    Retriever,
    VectorStore,
)
from aether.core.logging import get_logger

logger = get_logger("aether.container")


class MissingDependencyError(RuntimeError):
    """所选后端依赖缺失。"""


def _require(module: str, hint: str) -> None:
    try:
        __import__(module)
    except Exception as exc:
        raise MissingDependencyError(
            f"后端依赖缺失：{module}（{exc}）。安装方式：{hint}"
        ) from exc


def build_embedding(cfg: Config) -> EmbeddingModel:
    if cfg.embedding_backend == "tfidf":
        from aether.adapters.embeddings.tfidf_embedding import TfidfEmbedding

        return TfidfEmbedding(dim=cfg.embedding_dim)
    if cfg.embedding_backend == "sentence_transformer":
        _require("sentence_transformers", "pip install sentence-transformers")
        from aether.adapters.embeddings.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        return SentenceTransformerEmbedding(model_name=cfg.embedding_model)
    raise ValueError(f"未知 embedding 后端：{cfg.embedding_backend}")


def build_vectorstore(cfg: Config) -> VectorStore:
    if cfg.vectorstore_backend == "memory":
        from aether.adapters.vectorstore.memory_vector_store import MemoryVectorStore

        return MemoryVectorStore()
    if cfg.vectorstore_backend == "chroma":
        _require("chromadb", "pip install chromadb")
        from aether.adapters.vectorstore.chroma_vector_store import ChromaVectorStore

        return ChromaVectorStore()
    if cfg.vectorstore_backend == "faiss":
        _require("faiss", "pip install faiss-cpu")
        from aether.adapters.vectorstore.faiss_vector_store import FaissVectorStore

        return FaissVectorStore(dim=cfg.embedding_dim)
    raise ValueError(f"未知 vectorstore 后端：{cfg.vectorstore_backend}")


def build_llm(cfg: Config) -> LLMBackend:
    if cfg.llm_backend == "mock":
        from aether.adapters.llm.mock_llm import MockLLM

        return MockLLM(model=cfg.llm_model)
    if cfg.llm_backend == "openai_compatible":
        _require("httpx", "pip install httpx")
        from aether.adapters.llm.openai_compatible_llm import OpenAICompatibleLLM

        return OpenAICompatibleLLM(
            base_url=cfg.llm_base_url,
            api_key=cfg.llm_api_key,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
        )
    if cfg.llm_backend == "local":
        _require("ctransformers", "pip install ctransformers")
        from aether.adapters.llm.local_llm import LocalLLM

        return LocalLLM(model_path=cfg.llm_model)
    raise ValueError(f"未知 llm 后端：{cfg.llm_backend}")


def build_retriever(
    cfg: Config, embedding: EmbeddingModel, vectorstore: VectorStore
) -> Retriever:
    from aether.pipeline.retrieval import DefaultRetriever

    return DefaultRetriever(embedding=embedding, vectorstore=vectorstore, top_k=cfg.top_k)


def build_ingestor(
    cfg: Config, embedding: EmbeddingModel, vectorstore: VectorStore
) -> Ingestor:
    from aether.pipeline.ingest import DefaultIngestor

    return DefaultIngestor(
        embedding=embedding,
        vectorstore=vectorstore,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )


def build_rag(cfg: Config, retriever: Retriever, llm: LLMBackend):
    from aether.pipeline.rag import RAGPipeline

    return RAGPipeline(retriever=retriever, llm=llm, model=cfg.llm_model)


def build_agent(cfg: Config, llm: LLMBackend, retriever: Retriever) -> Agent:
    from aether.agent.react_agent import ReActAgent
    from aether.agent.tools.calculator import CalculatorTool
    from aether.agent.tools.retriever_tool import RetrieverTool

    tools = [CalculatorTool(), RetrieverTool(retriever)]
    return ReActAgent(llm=llm, tools=tools, model=cfg.llm_model)


def build_container(cfg: Config) -> dict[str, Any]:
    """装配完整系统，返回命名组件字典。"""

    embedding = build_embedding(cfg)
    vectorstore = build_vectorstore(cfg)
    llm = build_llm(cfg)
    retriever = build_retriever(cfg, embedding, vectorstore)
    ingestor = build_ingestor(cfg, embedding, vectorstore)
    rag = build_rag(cfg, retriever, llm)
    agent = build_agent(cfg, llm, retriever)

    components: dict[str, Any] = {
        "config": cfg,
        "embedding": embedding,
        "vectorstore": vectorstore,
        "llm": llm,
        "retriever": retriever,
        "ingestor": ingestor,
        "rag": rag,
        "agent": agent,
    }
    logger.info(
        "容器装配完成：embedding=%s vectorstore=%s llm=%s",
        cfg.embedding_backend,
        cfg.vectorstore_backend,
        cfg.llm_backend,
    )
    return components
