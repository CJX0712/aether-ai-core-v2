# 使用指南（aether-ai-core）

作者：晨星

## 1. 命令行（CLI）

```bash
# 摄入：文本或文件（txt/md/json/pdf）
python -m aether ingest --text "晨星是本项目作者。"
python -m aether ingest --file ./docs/architecture.md

# RAG 问答
python -m aether query --query "谁是作者？" --top_k 3

# Agent（自动路由检索 / 计算工具）
python -m aether chat --task "aether-ai-core 采用什么设计？"
python -m aether chat --task "计算 (128+64)*3"

# 服务与状态
python -m aether serve
python -m aether health
python -m aether eval        # 一键端到端自检
```

## 2. HTTP API

```bash
# 健康检查
curl http://localhost:8000/health

# 摄入
curl -X POST http://localhost:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text":"晨星是本项目作者。","source":"cli"}'

# RAG 问答
curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"谁是作者？"}'

# Agent
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"task":"计算 12*8+3"}'
```

响应示例（`/v1/query`）：
```json
{
  "answer": "晨星是本项目作者。",
  "sources": [{"chunk": {"id": "...", "text": "..."}, "score": 0.82}],
  "model": "mock"
}
```

## 3. Python API

```python
from aether.core.config import load_config
from aether.core.container import build_container

comp = build_container(load_config())
comp["ingestor"].ingest([Document(id="d1", content="晨星是作者。", source="x")])

# RAG
res = comp["rag"].ask("谁是作者？")
print(res.answer, [s.chunk.text for s in res.sources])

# Agent
out = comp["agent"].run("aether-ai-core 的作者还负责什么？")
print(out.answer, [s.tool for s in out.steps])
```

> 注：以上示例在默认栈（mock）下即可运行，无需 GPU。

## 4. 自定义后端

只需实现对应接口并在 `core/container.py` 注册，业务调用保持一致：

```python
from aether.core.interfaces import EmbeddingModel

class MyEmbedding(EmbeddingModel):
    @property
    def dim(self): return 768
    def embed(self, texts): ...
```
