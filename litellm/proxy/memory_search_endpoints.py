"""Semantic search over the session store for API callers.

POST /v1/memory/search  {"query": "...", "limit": 10, "agent": "...", "harness": "...", "user": "...", "team": "..."}

Results are always limited to what the caller may read. A proxy admin may search everything and narrow it by
user or team. A key that belongs to a user searches every session written by that user's keys (and, when
MEMORY_MASTER_KEY_USER names that user, the master key's sessions too). A key without a user only searches the
sessions it wrote itself. Agent and harness narrow any of those further.

Search runs litellm/proxy/memory_vectors.py against DATABASE_URL, embedding and reranking through DeepInfra with
DEEPINFRA_API_KEY; the endpoint answers 503 when either is missing.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Annotated, Final

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.memory_vectors import DeepInfra, Hit, Scope, search

if TYPE_CHECKING:
    import psycopg

router: Final = APIRouter()

MAX_LIMIT: Final = 50


class MemorySearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=10, ge=1, le=MAX_LIMIT)
    user: str | None = None
    team: str | None = None
    agent: str | None = None
    harness: str | None = None


class MemoryHit(TypedDict):
    score: ReadOnly[float]
    session_id: ReadOnly[str]
    seq: ReadOnly[int]
    title: ReadOnly[str | None]
    source: ReadOnly[str | None]
    agent: ReadOnly[str | None]
    harness: ReadOnly[str | None]
    chunk: ReadOnly[int]
    content: ReadOnly[str]


class MemorySearchResponse(TypedDict):
    object: ReadOnly[str]
    data: ReadOnly[tuple[MemoryHit, ...]]


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=403, detail=detail)


def caller_scope(caller: UserAPIKeyAuth, request: MemorySearchRequest, master_key_user: str | None) -> Scope:
    """The sessions this caller may search, narrowed by what the request asks for"""
    if caller.user_role == LitellmUserRoles.PROXY_ADMIN:
        return Scope(
            user=request.user,
            team=request.team,
            agent=request.agent,
            harness=request.harness,
            master_key_user=master_key_user,
        )
    if caller.user_id:
        if request.user not in (None, caller.user_id):
            raise _forbidden("Only a proxy admin may search another user's sessions")
        if request.team not in (None, caller.team_id):
            raise _forbidden("A key may only narrow a search to its own team")
        return Scope(
            user=caller.user_id,
            team=request.team,
            agent=request.agent,
            harness=request.harness,
            master_key_user=master_key_user,
        )
    if request.user is not None or request.team is not None:
        raise _forbidden("This key has no user, so it can only search the sessions it wrote")
    if not caller.api_key:
        raise HTTPException(status_code=401, detail="An API key is required to search memory")
    return Scope(owner=caller.api_key, agent=request.agent, harness=request.harness)


def _hit(hit: Hit) -> MemoryHit:
    found: Final[MemoryHit] = {
        "score": hit.score,
        "session_id": hit.session_id,
        "seq": hit.seq,
        "title": hit.title,
        "source": hit.source,
        "agent": hit.agent,
        "harness": hit.harness,
        "chunk": hit.chunk_no,
        "content": hit.content,
    }
    return found


def _run(database_url: str, api_key: str, query: str, scope: Scope, limit: int) -> tuple[Hit, ...]:
    import psycopg

    deepinfra: Final = DeepInfra(client=httpx.Client(timeout=60), api_key=api_key)
    conn: psycopg.Connection
    with deepinfra.client, psycopg.connect(database_url, autocommit=True) as conn:
        return search(conn, deepinfra.embed, deepinfra.rerank, query, scope=scope, limit=limit)


@router.post("/v1/memory/search", tags=["memory"])  # mutable-ok: FastAPI takes tags as a list
async def memory_search(
    request: MemorySearchRequest, user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)]
) -> MemorySearchResponse:
    database_url: Final = os.environ.get("DATABASE_URL")
    api_key: Final = os.environ.get("DEEPINFRA_API_KEY")
    if not database_url or not api_key:
        raise HTTPException(status_code=503, detail="Memory search needs DATABASE_URL and DEEPINFRA_API_KEY")
    scope: Final = caller_scope(user_api_key_dict, request, os.environ.get("MEMORY_MASTER_KEY_USER") or None)
    hits: Final = await asyncio.to_thread(_run, database_url, api_key, request.query, scope, request.limit)
    response: Final[MemorySearchResponse] = {"object": "list", "data": tuple(_hit(hit) for hit in hits)}
    return response
