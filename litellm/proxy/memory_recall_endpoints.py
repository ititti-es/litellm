"""REST companions to POST /v1/memory/search, for clients that blend recall with sources of their own.

POST /v1/memory/rerank   {"query": "...", "documents": ["...", ...]}
    Scores each document against the query with the same Qwen3 reranker memory search uses, in input order, so a
    client can rank its own candidates (workspace notes, files) on the scale of memory search hits. It reads nothing
    from the store.
POST /v1/memory/context  {"session_id": "...", "seq": 12, "before": 4, "after": 4}
    The messages around a memory search hit, scoped to the caller exactly like the search that found it.

Both answer 503 when DEEPINFRA_API_KEY (rerank) or DATABASE_URL (context) is missing.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory_mcp_tools import ContextArgs, ContextMissing, ContextWindow, context_window
from litellm.proxy.memory_search_endpoints import MemorySearchRequest, caller_scope
from litellm.proxy.memory_vectors import DeepInfra, Scope, psycopg_url

if TYPE_CHECKING:
    import psycopg

router: Final = APIRouter()

MAX_DOCUMENTS: Final = 64
MAX_DOCUMENT_CHARS: Final = 8000


class RerankRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    documents: tuple[Annotated[str, Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)], ...] = Field(
        min_length=1, max_length=MAX_DOCUMENTS
    )


class RerankScore(TypedDict):
    index: ReadOnly[int]
    score: ReadOnly[float]


class RerankResponse(TypedDict):
    object: ReadOnly[str]
    data: ReadOnly[tuple[RerankScore, ...]]


class ContextRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    before: int = Field(default=4, ge=0, le=20)
    after: int = Field(default=4, ge=0, le=20)


class ContextMessage(TypedDict):
    seq: ReadOnly[int]
    text: ReadOnly[str]
    hit: ReadOnly[bool]


class ContextResponse(TypedDict):
    object: ReadOnly[str]
    session_id: ReadOnly[str]
    title: ReadOnly[str | None]
    started: ReadOnly[str | None]
    harness: ReadOnly[str | None]
    agent: ReadOnly[str | None]
    last_seq: ReadOnly[int | None]
    items: ReadOnly[tuple[ContextMessage, ...]]


def _rerank(api_key: str, request: RerankRequest) -> tuple[float, ...]:
    with httpx.Client(timeout=60) as client:
        return DeepInfra(client=client, api_key=api_key).rerank(request.query, request.documents)


def _context(database_url: str, request: ContextRequest, scope: Scope) -> ContextWindow:
    import psycopg

    conn: psycopg.Connection
    with psycopg.connect(psycopg_url(database_url), autocommit=True) as conn:
        return context_window(
            conn,
            ContextArgs(session_id=request.session_id, seq=request.seq, before=request.before, after=request.after),
            scope,
        )


@router.post("/v1/memory/rerank", tags=["memory"])  # mutable-ok: FastAPI takes tags as a list
async def memory_rerank(
    request: RerankRequest, user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> RerankResponse:
    _ = user_api_key_dict  # any valid key may rank its own candidates; the store is not read
    api_key: Final = os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="Memory rerank needs DEEPINFRA_API_KEY")
    scores: Final = await asyncio.to_thread(_rerank, api_key, request)
    response: Final[RerankResponse] = {
        "object": "list",
        "data": tuple({"index": index, "score": score} for index, score in enumerate(scores)),
    }
    return response


@router.post("/v1/memory/context", tags=["memory"])  # mutable-ok: FastAPI takes tags as a list
async def memory_context(
    request: ContextRequest, user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> ContextResponse:
    database_url: Final = os.environ.get("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="Memory context needs DATABASE_URL")
    scope: Final = caller_scope(
        user_api_key_dict, MemorySearchRequest(query="-"), os.environ.get("MEMORY_MASTER_KEY_USER") or None
    )
    try:
        window: Final = await asyncio.to_thread(_context, database_url, request, scope)
    except ContextMissing as missing:
        raise HTTPException(status_code=404, detail=str(missing)) from missing
    session: Final = window.session
    response: Final[ContextResponse] = {
        "object": "memory.context",
        "session_id": session.session_id,
        "title": session.title,
        "started": session.created_at.isoformat() if session.created_at else None,
        "harness": session.harness,
        "agent": session.agent,
        "last_seq": window.last_seq,
        "items": tuple({"seq": item.seq, "text": item.text, "hit": item.hit} for item in window.items),
    }
    return response
