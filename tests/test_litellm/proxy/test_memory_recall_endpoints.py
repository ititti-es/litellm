from datetime import datetime, timezone
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy import memory_recall_endpoints
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory_mcp_tools import (  # pyright: ignore[reportPrivateUsage]
    ContextItem,
    ContextMissing,
    ContextWindow,
    _Session,
)
from litellm.proxy.memory_vectors import Scope

PAGE: Final = UserAPIKeyAuth(api_key="hash_page", user_id="page", team_id="home")


def _client(caller: UserAPIKeyAuth = PAGE) -> TestClient:
    app: Final = FastAPI()
    app.include_router(memory_recall_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app)


def test_rerank_scores_documents_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_KEY", "key")
    seen: Final[list[tuple[str, tuple[str, ...]]]] = []  # mutable-ok: records the ranked request

    def fake_rerank(api_key: str, request: memory_recall_endpoints.RerankRequest) -> tuple[float, ...]:
        assert api_key == "key"
        seen.append((request.query, request.documents))
        return (0.25, 0.9)

    monkeypatch.setattr(memory_recall_endpoints, "_rerank", fake_rerank)
    response: Final = _client().post("/v1/memory/rerank", json={"query": "tomatoes", "documents": ["a", "b"]})
    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": [{"index": 0, "score": 0.25}, {"index": 1, "score": 0.9}]}
    assert seen == [("tomatoes", ("a", "b"))]


def test_rerank_refuses_empty_or_oversized_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPINFRA_API_KEY", "key")
    client: Final = _client()
    assert client.post("/v1/memory/rerank", json={"query": "q", "documents": []}).status_code == 422
    assert client.post("/v1/memory/rerank", json={"query": "q", "documents": ["x"] * 65}).status_code == 422
    monkeypatch.delenv("DEEPINFRA_API_KEY")
    assert client.post("/v1/memory/rerank", json={"query": "q", "documents": ["x"]}).status_code == 503


def _window() -> ContextWindow:
    session: Final = _Session(
        session_id="ses_1",
        version=0,
        title="Garden plans",
        created_at=datetime(2026, 9, 3, 8, tzinfo=timezone.utc),
        agent="sett",
        harness="opencode",
    )
    return ContextWindow(
        session=session,
        last_seq=9,
        items=(ContextItem(seq=3, text="user: when?", hit=False), ContextItem(seq=4, text="assistant: May", hit=True)),
    )


def test_context_answers_in_the_callers_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("MEMORY_MASTER_KEY_USER", "page")
    seen: Final[list[tuple[str, int, int, Scope]]] = []  # mutable-ok: records each read

    def fake_context(database_url: str, request: memory_recall_endpoints.ContextRequest, scope: Scope) -> ContextWindow:
        assert database_url == "postgresql://x"
        seen.append((request.session_id, request.seq, request.before, scope))
        return _window()

    monkeypatch.setattr(memory_recall_endpoints, "_context", fake_context)
    response: Final = _client().post("/v1/memory/context", json={"session_id": "ses_1", "seq": 4})
    assert response.status_code == 200
    body: Final = response.json()
    assert body["title"] == "Garden plans" and body["harness"] == "opencode" and body["last_seq"] == 9
    assert body["started"].startswith("2026-09-03")
    assert body["items"] == [
        {"seq": 3, "text": "user: when?", "hit": False},
        {"seq": 4, "text": "assistant: May", "hit": True},
    ]
    assert seen == [("ses_1", 4, 4, Scope(user="page", master_key_user="page"))]


def test_context_outside_the_callers_memory_is_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")

    def missing(database_url: str, request: memory_recall_endpoints.ContextRequest, scope: Scope) -> ContextWindow:
        raise ContextMissing(f"No session {request.session_id} in your memory.")

    monkeypatch.setattr(memory_recall_endpoints, "_context", missing)
    response: Final = _client().post("/v1/memory/context", json={"session_id": "ses_other", "seq": 0})
    assert response.status_code == 404
    assert "ses_other" in response.json()["detail"]
    assert _client().post("/v1/memory/context", json={"session_id": "s", "seq": 0, "after": 99}).status_code == 422
