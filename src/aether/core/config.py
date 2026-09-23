"""配置加载。

优先级：环境变量（AETHER_*）> YAML 文件 > 内置默认值。
任何模块都从 Config 读取参数，保证部署可配置、可复现。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "embedding": {"backend": "tfidf", "model": "tf-idf", "dim": 256},
    "vectorstore": {"backend": "memory"},
    "llm": {
        "backend": "mock",
        "base_url": "http://localhost:11434/v1",
        "api_key": "",
        "model": "mock",
        "temperature": 0.2,
        "max_tokens": 1024,
    },
    "retrieval": {"top_k": 5},
    "ingestion": {"chunk_size": 400, "chunk_overlap": 80},
    "server": {"host": "0.0.0.0", "port": 8000},
}


@dataclass
class Config:
    """全局配置。"""

    raw: dict[str, Any] = field(default_factory=dict)

    # embedding
    @property
    def embedding_backend(self) -> str:
        return self.raw["embedding"]["backend"]

    @property
    def embedding_model(self) -> str:
        return self.raw["embedding"]["model"]

    @property
    def embedding_dim(self) -> int:
        return int(self.raw["embedding"].get("dim", 256))

    # vectorstore
    @property
    def vectorstore_backend(self) -> str:
        return self.raw["vectorstore"]["backend"]

    # llm
    @property
    def llm_backend(self) -> str:
        return self.raw["llm"]["backend"]

    @property
    def llm_base_url(self) -> str:
        return self.raw["llm"]["base_url"]

    @property
    def llm_api_key(self) -> str:
        return self.raw["llm"]["api_key"]

    @property
    def llm_model(self) -> str:
        return self.raw["llm"]["model"]

    @property
    def llm_temperature(self) -> float:
        return float(self.raw["llm"]["temperature"])

    @property
    def llm_max_tokens(self) -> int:
        return int(self.raw["llm"]["max_tokens"])

    # retrieval
    @property
    def top_k(self) -> int:
        return int(self.raw["retrieval"]["top_k"])

    # ingestion
    @property
    def chunk_size(self) -> int:
        return int(self.raw["ingestion"]["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return int(self.raw["ingestion"]["chunk_overlap"])

    # server
    @property
    def server_host(self) -> str:
        return self.raw["server"]["host"]

    @property
    def server_port(self) -> int:
        return int(self.raw["server"]["port"])

    def section(self, name: str) -> dict[str, Any]:
        return self.raw[name]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None = None) -> Config:
    """加载配置：文件（可选）叠加环境变量覆盖。"""

    raw = _deep_merge(DEFAULTS, {})
    cfg_path = path or os.environ.get("AETHER_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})

    env_map = {
        "AETHER_EMBEDDING_BACKEND": ("embedding", "backend"),
        "AETHER_EMBEDDING_MODEL": ("embedding", "model"),
        "AETHER_VECTORSTORE_BACKEND": ("vectorstore", "backend"),
        "AETHER_LLM_BACKEND": ("llm", "backend"),
        "AETHER_LLM_BASE_URL": ("llm", "base_url"),
        "AETHER_LLM_API_KEY": ("llm", "api_key"),
        "AETHER_LLM_MODEL": ("llm", "model"),
        "AETHER_TOP_K": ("retrieval", "top_k"),
        "AETHER_SERVER_PORT": ("server", "port"),
    }
    for env_key, (sec, key) in env_map.items():
        if env_key in os.environ:
            raw[sec][key] = os.environ[env_key]

    return Config(raw=raw)
