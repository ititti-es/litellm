from datetime import datetime, timezone
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy import memory_mcp_tools
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.memory_mcp_tools import (
    MEMORY_CONTEXT_TOOL_NAME,
    MEMORY_SEARCH_TOOL_NAME,
    MEMORY_TOOL_DEFINITIONS,
    MEMORY_TOOL_NAMES,
    call_memory_tool,
    format_hits,
    recall_scope,
)
from litellm.proxy.memory_vectors import Hit, Scope

MASTER: Final = UserAPIKeyAuth(api_key="litellm_proxy_master_key", user_role=LitellmUserRoles.PROXY_ADMIN)
ADMIN_PAGE: Final = UserAPIKeyAuth(api_key="hash_admin", user_id="page", user_role=LitellmUserRoles.PROXY_ADMIN)
PAGE: Final = UserAPIKeyAuth(api_key="hash_page", user_id="page", team_id="home")
KEYLESS_USER: Final = UserAPIKeyAuth(api_key="hash_bot")


def test_the_master_key_recalls_its_humans_sessions_not_everyones() -> None:
    assert recall_scope(MASTER, None, None, "page") == Scope(user="page", master_key_user="page")
    assert recall_scope(MASTER, "sett", "openclaw", None) == Scope(
        owner="litellm_proxy_master_key", agent="sett", harness="openclaw"
    )


def test_an_admin_key_with_a_user_recalls_that_users_sessions() -> None:
    assert recall_scope(ADMIN_PAGE, None, "opencode", "someone") == Scope(
        user="page", harness="opencode", master_key_user="someone"
    )


def test_user_and_keyless_keys_keep_the_endpoints_scope() -> None:
    assert recall_scope(PAGE, "sett", None, "page") == Scope(user="page", agent="sett", master_key_user="page")
    assert recall_scope(KEYLESS_USER, None, None, "page") == Scope(owner="hash_bot")


def test_an_admin_without_any_identity_is_refused() -> None:
    with pytest.raises(HTTPException) as refused:
        _ = recall_scope(UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), None, None, None)
    assert refused.value.status_code == 401


def test_definitions_name_both_tools_and_require_their_keys() -> None:
    by_name: Final = {definition["name"]: definition for definition in MEMORY_TOOL_DEFINITIONS}
    assert set(by_name) == MEMORY_TOOL_NAMES
    assert by_name[MEMORY_SEARCH_TOOL_NAME]["inputSchema"]["required"] == ["query"]
    assert by_name[MEMORY_CONTEXT_TOOL_NAME]["inputSchema"]["required"] == ["session_id", "seq"]
    assert "memory_context" in by_name[MEMORY_SEARCH_TOOL_NAME]["description"]


def _hit(**fields: object) -> Hit:
    base: Final = {
        "score": 0.875,
        "session_id": "ses_1",
        "seq": 12,
        "title": "Garden plans",
        "source": "opencode",
        "agent": "sett",
        "harness": "opencode",
        "chunk_no": 0,
        "content": "user: plant the tomatoes in May",
    }
    return Hit(**{**base, **fields})  # pyright: ignore[reportArgumentType]


def test_hits_render_with_the_ids_memory_context_needs() -> None:
    started: Final = {"ses_1": datetime(2026, 9, 3, 8, tzinfo=timezone.utc)}
    text: Final = format_hits((_hit(), _hit(session_id="ses_2", title=None, content="x" * 2000)), started)
    assert text.startswith("[1] session_id=ses_1 seq=12 score=0.88 started=2026-09-03 harness=opencode agent=sett")
    assert "title: Garden plans\nuser: plant the tomatoes in May" in text
    assert "[2] session_id=ses_2" in text and "started=unknown date" in text and "title: (untitled)" in text
    assert text.endswith(" [...]")
    assert format_hits((), {}).startswith("No matching memories")


@pytest.mark.asyncio
async def test_search_runs_in_the_callers_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Final[list[tuple[str, int, Scope]]] = []  # mutable-ok: records each search

    def fake_search(args: memory_mcp_tools.SearchArgs, scope: Scope) -> str:
        seen.append((args.query, args.limit, scope))
        return "found"

    monkeypatch.setattr(memory_mcp_tools, "_search", fake_search)
    monkeypatch.setenv("MEMORY_MASTER_KEY_USER", "page")
    result: Final = await call_memory_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "tomatoes", "harness": "opencode"}, PAGE)
    assert not result.isError
    assert result.content[0].text == "found"
    assert seen == [("tomatoes", 8, Scope(user="page", harness="opencode", master_key_user="page"))]


@pytest.mark.asyncio
async def test_context_runs_in_the_callers_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: Final[list[tuple[str, int, int, Scope]]] = []  # mutable-ok: records each read

    def fake_context(args: memory_mcp_tools.ContextArgs, scope: Scope) -> str:
        seen.append((args.session_id, args.seq, args.before, scope))
        return "window"

    monkeypatch.setattr(memory_mcp_tools, "_context", fake_context)
    monkeypatch.delenv("MEMORY_MASTER_KEY_USER", raising=False)
    result: Final = await call_memory_tool(MEMORY_CONTEXT_TOOL_NAME, {"session_id": "ses_1", "seq": 3}, KEYLESS_USER)
    assert result.content[0].text == "window"
    assert seen == [("ses_1", 3, 4, Scope(owner="hash_bot"))]


@pytest.mark.asyncio
async def test_bad_calls_come_back_as_tool_errors() -> None:
    missing_query: Final = await call_memory_tool(MEMORY_SEARCH_TOOL_NAME, {}, PAGE)
    assert missing_query.isError and "Invalid arguments" in missing_query.content[0].text
    widened: Final = await call_memory_tool(MEMORY_CONTEXT_TOOL_NAME, {"session_id": "s", "seq": 0, "after": 99}, PAGE)
    assert widened.isError
    anonymous: Final = await call_memory_tool(MEMORY_SEARCH_TOOL_NAME, {"query": "q"}, None)
    assert anonymous.isError and "authenticated" in anonymous.content[0].text


def test_context_gives_tool_output_a_smaller_share_than_conversation() -> None:
    said: Final = memory_mcp_tools._item_text({"type": "message", "role": "user", "content": "y" * 5000})
    ran: Final = memory_mcp_tools._item_text({"type": "function_call_output", "call_id": "c", "output": "z" * 5000})
    assert len(said) > 3000 > 1000 > len(ran)
    assert memory_mcp_tools._item_text({"type": "reasoning"}).startswith("(no text")
