"""共享 fixtures：用默认（mock）栈构建容器并预摄入样例语料。"""

from __future__ import annotations

import pytest

from aether.core.config import load_config
from aether.core.container import build_container
from aether.core.models import Document

SAMPLE = [
    Document(id="d1", content="晨星是 aether-ai-core 项目的作者，负责整体架构与文档。", source="intro"),
    Document(id="d2", content="该系统采用接口驱动的模块化设计，embedding、向量库、大模型均可插拔替换。", source="design"),
    Document(id="d3", content="默认栈无需 GPU 即可运行，世界顶级后端以可选适配器形式接入。", source="deploy"),
]


@pytest.fixture
def components():
    comp = build_container(load_config())
    comp["ingestor"].ingest(SAMPLE)
    return comp
