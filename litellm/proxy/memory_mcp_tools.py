"""Memory recall as tools on liminal's own MCP server (/mcp): memory_search and memory_context.

Every authenticated MCP caller sees both tools next to its upstream servers' tools, so any harness that already
connects to /mcp (OpenClaw, opencode) gets recall without new config. Both run in the caller's scope: a key that
belongs to a user recalls that user's sessions, a key without a user recalls only what it wrote (see
memory_search_endpoints.caller_scope). A proxy admin key is narrowed the same way rather than searching everyone:
it recalls the sessions of the user it belongs to, or of MEMORY_MASTER_KEY_USER for the master key, and otherwise
only its own. MCP callers are agents, and an agent's recall should be its own human's memory.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final

import httpx
from fastapi import HTTPException
from psycopg.rows import class_row
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.memory_search_endpoints import MemorySearchRequest, caller_scope
from litellm.proxy.memory_vectors import (
    _SESSION_IN_SCOPE,
    ITEM_CONTENT,
    DeepInfra,
    Hit,
    Scope,
    labeled_text,
    psycopg_url,
    search,
)

if TYPE_CHECKING:
    import psycopg
    from mcp.types import CallToolResult

MEMORY_SEARCH_TOOL_NAME: Final = "memory_search"
MEMORY_CONTEXT_TOOL_NAME: Final = "memory_context"
MEMORY_TOOL_NAMES: Final = frozenset((MEMORY_SEARCH_TOOL_NAME, MEMORY_CONTEXT_TOOL_NAME))

HIT_CHARS: Final = 1500
CONTEXT_ITEM_CHARS: Final = 3000
CONTEXT_TOOL_OUTPUT_CHARS: Final = 800
"""Tool output is mostly noise next to what was said, so it gets a much smaller share of the window"""
MAX_CONTEXT_SIDE: Final = 20

_FILTERS: Final = {
    "agent": {
        "type": "string",
        "description": "Only recall sessions of this agent (for example sett). Omit to search every agent.",
    },
    "harness": {
        "type": "string",
        "description": "Only recall sessions from this harness (openclaw, opencode, claude_code). Omit to search all.",
    },
}

MEMORY_TOOL_DEFINITIONS: Final = (
    {
        "name": MEMORY_SEARCH_TOOL_NAME,
        "description": (
            "Search long-term memory: every past conversation with this user, across all sessions, agents and "
            "harnesses (OpenClaw, opencode, Claude Code), not only the current chat. Use it before answering "
            'whenever the user refers to anything from before this conversation ("remember", "last time", '
            '"what did we decide", "that thing I mentioned"), asks about their own preferences, projects, '
            "people, plans or past decisions, or when you would otherwise guess or ask them to repeat themselves. "
            "It is semantic search with reranking, so describe what you are looking for in plain words and include "
            "distinctive names or terms; try a rephrased query if the first one misses. Returns the best matching "
            "passages, each with its session_id and seq. Call memory_context with those to read the conversation "
            "around a hit. Results only ever include this user's own sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall, in natural language."},
                "limit": {
                    "type": "integer",
                    "description": "How many passages to return (1 to 20).",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 20,
                },
                **_FILTERS,
            },
            "required": ["query"],  # mutable-ok: JSON schema
        },
    },
    {
        "name": MEMORY_CONTEXT_TOOL_NAME,
        "description": (
            "Read the conversation around a memory_search hit: the messages just before and after it in that past "
            "session, in order, with the session's title and date. Use it when a hit looks relevant but the passage "
            "alone does not answer the question, or to see what was decided after it. Pass session_id and seq "
            "exactly as memory_search returned them; raise before/after to read further."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "The session_id of a memory_search hit."},
                "seq": {"type": "integer", "description": "The seq of the memory_search hit."},
                "before": {
                    "type": "integer",
                    "description": "Messages to include before the hit (0 to 20).",
                    "default": 4,
                    "minimum": 0,
                    "maximum": MAX_CONTEXT_SIDE,
                },
                "after": {
                    "type": "integer",
                    "description": "Messages to include after the hit (0 to 20).",
                    "default": 4,
                    "minimum": 0,
                    "maximum": MAX_CONTEXT_SIDE,
                },
            },
            "required": ["session_id", "seq"],  # mutable-ok: JSON schema
        },
    },
)


class SearchArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)
    agent: str | None = None
    harness: str | None = None


class ContextArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    session_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    before: int = Field(default=4, ge=0, le=MAX_CONTEXT_SIDE)
    after: int = Field(default=4, ge=0, le=MAX_CONTEXT_SIDE)


def recall_scope(caller: UserAPIKeyAuth, agent: str | None, harness: str | None, master_key_user: str | None) -> Scope:
    """The sessions an MCP caller may recall; raises HTTPException when the key may not search at all"""
    if caller.user_role != LitellmUserRoles.PROXY_ADMIN:
        return caller_scope(caller, MemorySearchRequest(query="-", agent=agent, harness=harness), master_key_user)
    user: Final = caller.user_id or master_key_user
    if user:
        return Scope(user=user, agent=agent, harness=harness, master_key_user=master_key_user)
    if not caller.api_key:
        raise HTTPException(status_code=401, detail="An API key is required to search memory")
    return Scope(owner=caller.api_key, agent=agent, harness=harness)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _day(at: datetime | None) -> str:
    return at.date().isoformat() if at is not None else "unknown date"


def format_hits(hits: tuple[Hit, ...], started: Mapping[str, datetime]) -> str:
    if not hits:
        return "No matching memories. Try different words, or a broader description."
    blocks: Final = tuple(
        f"[{number}] session_id={hit.session_id} seq={hit.seq} score={hit.score:.2f} "
        f"started={_day(started.get(hit.session_id))} harness={hit.harness or '-'} agent={hit.agent or '-'}\n"
        f"title: {hit.title or '(untitled)'}\n{_clip(hit.content.strip(), HIT_CHARS)}"
        for number, hit in enumerate(hits, start=1)
    )
    return "\n\n".join(blocks)


_SESSION_SQL: Final = f"""
SELECT s.session_id, s.version, s.title, s.created_at,
       COALESCE(s.metadata->>'agent', v.metadata->>'agent') AS agent,
       COALESCE(s.metadata->>'harness', s.source, v.metadata->>'harness') AS harness
FROM "LiteLLM_ConversationSession" s
LEFT JOIN "LiteLLM_VerificationToken" v ON v.token = s.owner
WHERE s.session_id = ANY(%(session_ids)s) AND {_SESSION_IN_SCOPE}
"""

_HIT_VERSION_SQL: Final = """
SELECT COALESCE(
    (SELECT version FROM "LiteLLM_ConversationItem"
     WHERE session_id = %(session_id)s AND version = %(version)s AND seq = %(seq)s),
    (SELECT max(version) FROM "LiteLLM_ConversationItem" WHERE session_id = %(session_id)s AND seq = %(seq)s)
) AS version
"""

_WINDOW_SQL: Final = """
SELECT seq, content, (SELECT max(seq) FROM "LiteLLM_ConversationItem"
                      WHERE session_id = %(session_id)s AND version = %(version)s) AS last_seq
FROM "LiteLLM_ConversationItem"
WHERE session_id = %(session_id)s AND version = %(version)s AND seq BETWEEN %(low)s AND %(high)s
ORDER BY seq
"""


@dataclass(frozen=True, slots=True)
class _Session:
    session_id: str
    version: int
    title: str | None
    created_at: datetime | None
    agent: str | None
    harness: str | None


@dataclass(frozen=True, slots=True)
class _WindowItem:
    seq: int
    content: object
    last_seq: int


def _sessions(conn: psycopg.Connection, session_ids: tuple[str, ...], scope: Scope) -> Mapping[str, _Session]:
    with conn.cursor(row_factory=class_row(_Session)) as cursor:
        rows: Final = cursor.execute(
            _SESSION_SQL,
            {**scope.params(), "session_ids": list(session_ids)},  # mutable-ok: psycopg adapts only a list to an array
        ).fetchall()
    return {row.session_id: row for row in rows}


def _item_text(content: object) -> str:
    parsed: Final = ITEM_CONTENT.validate_python(content)
    text: Final = labeled_text(parsed) if parsed is not None else None
    if text is None:
        return "(no text: image, reasoning or empty item)"
    is_tool_output: Final = parsed is not None and parsed.get("type") == "function_call_output"
    return _clip(text, CONTEXT_TOOL_OUTPUT_CHARS if is_tool_output else CONTEXT_ITEM_CHARS)


class ContextMissing(LookupError):
    """The session or message a context read asked for is not in the caller's memory"""


@dataclass(frozen=True, slots=True)
class ContextItem:
    seq: int
    text: str
    hit: bool


@dataclass(frozen=True, slots=True)
class ContextWindow:
    session: _Session
    last_seq: int | None
    items: tuple[ContextItem, ...]


def context_window(conn: psycopg.Connection, args: ContextArgs, scope: Scope) -> ContextWindow:
    session: Final = _sessions(conn, (args.session_id,), scope).get(args.session_id)
    if session is None:
        raise ContextMissing(f"No session {args.session_id} in your memory.")
    with conn.cursor() as cursor:
        found: Final = cursor.execute(
            _HIT_VERSION_SQL, {"session_id": session.session_id, "version": session.version, "seq": args.seq}
        ).fetchone()
    version: Final = found[0] if found else None
    if version is None:
        raise ContextMissing(f"Session {args.session_id} has no message at seq {args.seq}.")
    with conn.cursor(row_factory=class_row(_WindowItem)) as cursor:
        window: Final = cursor.execute(
            _WINDOW_SQL,
            {
                "session_id": session.session_id,
                "version": version,
                "low": args.seq - args.before,
                "high": args.seq + args.after,
            },
        ).fetchall()
    return ContextWindow(
        session=session,
        last_seq=window[-1].last_seq if window else None,
        items=tuple(
            ContextItem(seq=item.seq, text=_item_text(item.content), hit=item.seq == args.seq) for item in window
        ),
    )


def format_context(window: ContextWindow) -> str:
    session: Final = window.session
    header: Final = (
        f"session_id={session.session_id} title: {session.title or '(untitled)'}\n"
        f"started={_day(session.created_at)} harness={session.harness or '-'} agent={session.agent or '-'} "
        f"messages 0..{window.last_seq if window.last_seq is not None else '?'}"
    )
    lines: Final = tuple(f"{'>>' if item.hit else '  '} seq={item.seq} {item.text}" for item in window.items)
    return "\n".join((header, *lines))


def read_context(conn: psycopg.Connection, args: ContextArgs, scope: Scope) -> str:
    try:
        return format_context(context_window(conn, args, scope))
    except ContextMissing as missing:
        return str(missing)


def _environment() -> tuple[str, str]:
    database_url: Final = os.environ.get("DATABASE_URL")
    api_key: Final = os.environ.get("DEEPINFRA_API_KEY")
    if not database_url or not api_key:
        raise HTTPException(status_code=503, detail="Memory search needs DATABASE_URL and DEEPINFRA_API_KEY")
    return database_url, api_key


def _search(args: SearchArgs, scope: Scope) -> str:
    import psycopg

    database_url, api_key = _environment()
    deepinfra: Final = DeepInfra(client=httpx.Client(timeout=60), api_key=api_key)
    conn: psycopg.Connection
    with deepinfra.client, psycopg.connect(psycopg_url(database_url), autocommit=True) as conn:
        hits: Final = search(conn, deepinfra.embed, deepinfra.rerank, args.query, scope=scope, limit=args.limit)
        sessions: Final = _sessions(conn, tuple({hit.session_id for hit in hits}), scope)
    return format_hits(hits, {key: row.created_at for key, row in sessions.items() if row.created_at})


def _context(args: ContextArgs, scope: Scope) -> str:
    import psycopg

    database_url, _ = _environment()
    conn: psycopg.Connection
    with psycopg.connect(psycopg_url(database_url), autocommit=True) as conn:
        return read_context(conn, args, scope)


def _result(text: str, is_error: bool = False) -> CallToolResult:
    from mcp.types import CallToolResult, TextContent

    return CallToolResult(content=[TextContent(type="text", text=text)], isError=is_error)  # mutable-ok: MCP list


async def call_memory_tool(name: str, arguments: Mapping[str, object], caller: UserAPIKeyAuth | None) -> CallToolResult:
    """Run memory_search or memory_context for an MCP caller, answering errors as isError results"""
    if caller is None:
        return _result("Memory tools need an authenticated liminal key", is_error=True)
    master_key_user: Final = os.environ.get("MEMORY_MASTER_KEY_USER") or None
    try:
        if name == MEMORY_SEARCH_TOOL_NAME:
            search_args: Final = SearchArgs.model_validate(arguments)
            scope = recall_scope(caller, search_args.agent, search_args.harness, master_key_user)
            return _result(await asyncio.to_thread(_search, search_args, scope))
        context_args: Final = ContextArgs.model_validate(arguments)
        scope = recall_scope(caller, None, None, master_key_user)
        return _result(await asyncio.to_thread(_context, context_args, scope))
    except ValidationError as error:
        return _result(f"Invalid arguments for {name}: {error.errors(include_url=False)}", is_error=True)
    except HTTPException as error:
        return _result(f"Error: {error.detail}", is_error=True)
