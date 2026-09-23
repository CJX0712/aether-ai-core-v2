# 部署指南（aether-ai-core）

作者：晨星

## 1. 本地干净环境复现

```bash
git clone https://github.com/CJX0712/aether-ai-core-43873e.git && cd aether-ai-core-43873e
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # 锁定版本，可复现
# 或源码可编辑安装：
pip install -e .
python -m aether eval                     # 端到端自检
```

## 2. Docker（推荐）

```bash
docker build -t aether-ai-core .
docker run -p 8000:8000 aether-ai-core
```

健康检查：`GET /health` 返回 `{"status":"ok",...}`。Dockerfile 内置 `HEALTHCHECK`。

## 3. Docker Compose

```bash
cp .env.example .env   # 按需修改后端
docker compose up -d
curl http://localhost:8000/health
```

`docker-compose.yml` 已配置 `restart: unless-stopped` 与健康检查。

## 4. 环境变量（覆盖优先级最高）

| 变量 | 作用 | 默认 |
|------|------|------|
| `AETHER_EMBEDDING_BACKEND` | tfidf / sentence_transformer | tfidf |
| `AETHER_EMBEDDING_MODEL` | 模型名（sentence_transformers 时） | tf-idf |
| `AETHER_VECTORSTORE_BACKEND` | memory / chroma / faiss | memory |
| `AETHER_LLM_BACKEND` | mock / openai_compatible / local | mock |
| `AETHER_LLM_BASE_URL` | 兼容 API 地址（Ollama/vLLM） | http://localhost:11434/v1 |
| `AETHER_LLM_API_KEY` | API Key（云端） | 空 |
| `AETHER_LLM_MODEL` | 模型标识 | mock |
| `AETHER_TOP_K` | 检索条数 | 5 |
| `AETHER_SERVER_PORT` | 服务端口 | 8000 |

也可通过 YAML 配置文件（`AETHER_CONFIG=config.yaml`）集中管理。

## 5. 升级到世界顶级后端（生产）

1. 安装可选依赖：`pip install -e ".[world-class]"`
2. 配置环境变量指向本地 Ollama（`ollama run qwen2.5:7b`）或 vLLM 服务；
3. 重启服务，业务代码与接口不变。

## 6. 生产建议

- 向量库：`chroma` / `faiss` 替代 `memory` 以持久化与扩容；
- 推理：`openai_compatible` 指向 vLLM 集群，启用批处理与量化；
- 网关前置反向代理（Nginx）做鉴权与限流；
- 日志已结构化输出到 stderr，可接入集中采集。
