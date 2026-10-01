"""GET /v1/sessions/{id}/items: owner scoping, ordering and the store-disabled case."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy import session_endpoints
from litellm.proxy._types import UserAPIKeyAuth
from litellm.repositories.conversation_session_repository import StoredSession
from litellm.sessions.manager import client_session_key

CREATED = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _stored(owner: str, client_session_id: str) -> StoredSession:
    items = (
        SimpleNamespace(seq=1, is_output=True, created_at=CREATED, content='{"type": "message", "role": "assistant"}'),
        SimpleNamespace(seq=0, is_output=False, created_at=CREATED, content={"type": "message", "role": "user"}),
    )
    responses = (SimpleNamespace(response_id="resp_1", item_count=2),)
    return StoredSession(
        session_id=client_session_key(owner, client_session_id),
        owner=owner,
        version=3,
        last_upstream_at=CREATED,
        items=items,  # pyright: ignore[reportArgumentType]
        responses=responses,  # pyright: ignore[reportArgumentType]
    )


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    repo = MagicMock()
    repo.load_session = AsyncMock(return_value=None)
    monkeypatch.setattr(session_endpoints, "_repository", lambda: repo)
    return repo


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock()
    fake.flush = AsyncMock()
    monkeypatch.setattr(litellm, "session_manager", fake)
    return fake


def _client(api_key: str | None) -> TestClient:
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    app = FastAPI()
    app.include_router(session_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key=api_key)
    return TestClient(app)


def test_returns_items_in_order_with_response_boundaries(repository: MagicMock, manager: MagicMock) -> None:
    repository.load_session.return_value = _stored("hash-a", "oc-session-1")

    response = _client("hash-a").get("/v1/sessions/oc-session-1/items")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "oc-session-1"
    assert body["version"] == 3
    assert [item["seq"] for item in body["items"]] == [0, 1]
    assert body["items"][1]["content"] == {"type": "message", "role": "assistant"}
    assert body["responses"] == [{"id": "resp_1", "item_count": 2}]
    manager.flush.assert_awaited_once()
    repository.load_session.assert_awaited_once_with(client_session_key("hash-a", "oc-session-1"))


def test_another_key_cannot_read_the_session(repository: MagicMock, manager: MagicMock) -> None:
    repository.load_session.return_value = _stored("hash-a", "oc-session-1")

    response = _client("hash-b").get("/v1/sessions/oc-session-1/items")

    assert response.status_code == 404


def test_unknown_session_is_not_found(repository: MagicMock, manager: MagicMock) -> None:
    assert _client("hash-a").get("/v1/sessions/missing/items").status_code == 404


def test_disabled_store_is_not_found(repository: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "session_manager", None)

    assert _client("hash-a").get("/v1/sessions/oc-session-1/items").status_code == 404
    repository.load_session.assert_not_awaited()


def test_requires_an_api_key(repository: MagicMock, manager: MagicMock) -> None:
    assert _client(None).get("/v1/sessions/oc-session-1/items").status_code == 401


def test_post_maps_advertised_ids_onto_stored_responses(
    repository: MagicMock, manager: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository.load_session.return_value = _stored("hash-a", "oc-session-1")
    monkeypatch.setattr(session_endpoints, "stored_response_ids", lambda ids, caller: {"resp_1": ids[0]} if ids else {})

    response = _client("hash-a").post(
        "/v1/sessions/oc-session-1/items", json={"response_ids": ["resp_advertised_1", "resp_other"]}
    )

    assert response.status_code == 200
    assert response.json()["responses"] == [{"id": "resp_1", "item_count": 2, "client_id": "resp_advertised_1"}]


def test_plain_advertised_ids_map_to_the_provider_id() -> None:
    mapping = session_endpoints.stored_response_ids(["resp_plain"], UserAPIKeyAuth(api_key="hash-a"))

    assert mapping == {"resp_plain": "resp_plain"}
