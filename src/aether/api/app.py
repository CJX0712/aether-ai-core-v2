"""FastAPI 应用：统一网关入口。

端点：/health、/v1/ingest、/v1/query(RAG)、/v1/chat(Agent)。
组件在启动时按配置装配（见 core.container）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from aether import __version__
from aether.api.schemas import (
    ChatRequest,
    InfoResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
)
from aether.core.config import load_config
from aether.core.container import build_container
from aether.core.logging import get_logger
from aether.core.models import Document, HealthStatus

logger = get_logger("aether.api")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg = load_config()
    app.state.components = build_container(cfg)
    app.state.config = cfg
    logger.info("API 启动，组件已装配")
    yield
    app.state.components.clear()
    logger.info("API 关闭")


def create_app() -> FastAPI:
    app = FastAPI(title="aether-ai-core", version=__version__, lifespan=_lifespan)

    @app.get("/health", response_model=HealthStatus)
    def health():
        cfg = app.state.config
        return HealthStatus(
            status="ok",
            components={
                "embedding": cfg.embedding_backend,
                "vectorstore": cfg.vectorstore_backend,
                "llm": cfg.llm_backend,
            },
        )

    @app.get("/", response_model=InfoResponse)
    def info():
        cfg = app.state.config
        return InfoResponse(
            version=__version__,
            embedding=cfg.embedding_backend,
            vectorstore=cfg.vectorstore_backend,
            llm=cfg.llm_backend,
        )

    @app.post("/v1/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        ingestor = app.state.components["ingestor"]
        docs: list[Document] = []
        if req.documents:
            for d in req.documents:
                docs.append(Document(**d))
        if req.text:
            docs.append(Document(id="api", content=req.text, source=req.source))
        n = ingestor.ingest(docs)
        return IngestResponse(chunks=n)

    @app.post("/v1/query")
    def query(req: QueryRequest):
        rag = app.state.components["rag"]
        return rag.ask(req.query, top_k=req.top_k)

    @app.post("/v1/chat")
    def chat(req: ChatRequest):
        agent = app.state.components["agent"]
        return agent.run(req.task)

    return app


app = create_app()
