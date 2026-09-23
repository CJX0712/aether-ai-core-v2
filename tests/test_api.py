"""API 网关验证（FastAPI TestClient）。"""

import pytest
from fastapi.testclient import TestClient

from aether.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_query(client):
    r = client.post("/v1/ingest", json={"text": "晨星是本项目作者。", "source": "t"})
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/v1/query", json={"query": "作者是谁？"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert "sources" in body


def test_chat(client):
    r = client.post("/v1/chat", json={"task": "计算 12*8+3"})
    assert r.status_code == 200
    assert r.json()["steps"]
