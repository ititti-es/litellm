from datetime import datetime, timezone
from typing import Final

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from litellm.proxy import memory_search_endpoints
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory_search_endpoints import MemorySearchRequest, SearchOutcome, caller_scope
from litellm.proxy.memory_vectors import Hit, Scope

ADMIN: Final = UserAPIKeyAuth(api_key="litellm_proxy_master_key", user_role=LitellmUserRoles.PROXY_ADMIN)
PAGE: Final = UserAPIKeyAuth(api_key="hash_page", user_id="page", team_id="home")
KEYLESS_USER: Final = UserAPIKeyAuth(api_key="hash_bot")


def _request(**fields: object) -> MemorySearchRequest:
    return MemorySearchRequest.model_validate({"query": "q", **fields})


def test_an_admin_may_search_anyone_and_narrow_by_user_and_team() -> None:
    assert caller_scope(ADMIN, _request(), None) == Scope()
    assert caller_scope(ADMIN, _request(user="page", team="home", agent="sett"), "page") == Scope(
        user="page", team="home", agent="sett", master_key_user="page"
    )


def test_a_users_key_searches_all_of_that_users_sessions_and_nobody_elses() -> None:
    assert caller_scope(PAGE, _request(agent="sett", harness="openclaw"), "page") == Scope(
        user="page", agent="sett", harness="openclaw", master_key_user="page"
    )
    assert caller_scope(PAGE, _request(user="page", team="home"), None) == Scope(user="page", team="home")
    with pytest.raises(HTTPException) as other_user:
        _ = caller_scope(PAGE, _request(user="someone"), None)
    assert other_user.value.status_code == 403
    with pytest.raises(HTTPException) as other_team:
        _ = caller_scope(PAGE, _request(team="work"), None)
    assert other_team.value.status_code == 403


def test_a_key_without_a_user_only_searches_what_it_wrote() -> None:
    assert caller_scope(KEYLESS_USER, _request(agent="sett"), "page") == Scope(owner="hash_bot", agent="sett")
    with pytest.raises(HTTPException) as widened:
        _ = caller_scope(KEYLESS_USER, _request(user="page"), None)
    assert widened.value.status_code == 403


def _client(caller: UserAPIKeyAuth) -> TestClient:
    app: Final = FastAPI()
    app.include_router(memory_search_endpoints.router)
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    return TestClient(app)


def test_the_endpoint_searches_in_the_callers_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Final[list[Scope]] = []  # mutable-ok: records the scope the endpoint searched with

    def fake_run(database_url: str, api_key: str, query: str, scope: Scope, limit: int) -> SearchOutcome:
        seen.append(scope)
        hits = (
            Hit(
                score=0.9,
                session_id="ses_1",
                seq=4,
                title="garden",
                source=None,
                agent="sett",
                harness="openclaw",
                chunk_no=0,
                content="user: water the seedlings",
            ),
        )
        return SearchOutcome(hits, {"ses_1": datetime(2026, 9, 3, 8, tzinfo=timezone.utc)})

    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "unused")
    monkeypatch.setenv("MEMORY_MASTER_KEY_USER", "page")
    monkeypatch.setattr(memory_search_endpoints, "_run", fake_run)
    response: Final = _client(PAGE).post("/v1/memory/search", json={"query": "seedlings", "agent": "sett"})
    assert response.status_code == 200
    assert response.json()["data"][0] == {
        "score": 0.9,
        "session_id": "ses_1",
        "seq": 4,
        "title": "garden",
        "started": "2026-09-03T08:00:00+00:00",
        "source": None,
        "agent": "sett",
        "harness": "openclaw",
        "chunk": 0,
        "content": "user: water the seedlings",
    }
    assert seen == [Scope(user="page", agent="sett", master_key_user="page")]


def test_the_endpoint_is_unavailable_without_its_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    assert _client(PAGE).post("/v1/memory/search", json={"query": "x"}).status_code == 503


def test_the_endpoint_rejects_a_widened_scope_before_searching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "unused")
    monkeypatch.setattr(memory_search_endpoints, "_run", lambda *args: pytest.fail("searched"))
    assert _client(KEYLESS_USER).post("/v1/memory/search", json={"query": "x", "user": "page"}).status_code == 403
