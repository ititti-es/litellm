"""Read access to the gateway-owned session store.

GET  /v1/sessions/{client_session_id}/items
POST /v1/sessions/{client_session_id}/items  {"response_ids": [...]}

Returns the whole stored history of a conversation a client named with its session id (the
``litellm_session_id`` taken from ``x-litellm-session-id``, ``session_id`` and the other recognised
headers), so a client that delegates history to this gateway can show it without keeping its own copy.

The stored row is found the same way a turn finds it, by ``client_session_key(owner, client_session_id)``,
and the owner is the caller's hashed API key. A key therefore only ever reads conversations it wrote: any
other key derives a different row id, and a row whose owner does not match is reported as missing.

The store keys responses by the provider's own id, while a client only ever saw the id the gateway advertised
(possibly encrypted by the response-id security hook). POST names the advertised ids the client holds, and
each stored response it can be matched to carries that id back as ``client_id``, so the client can join its
own records to the stored turns without decoding gateway ids itself.

Compaction rebases a session onto a summary as a new version. Every version's items are returned, each with its
``version``, and each response carries the version its ``item_count`` counts in. Items that a compaction wrote
(the summary and any lead-in before it) are flagged ``summary`` so a client showing the conversation can leave
them out; the conversation they replaced is still there under the earlier version.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import litellm
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
from litellm.repositories.conversation_session_repository import ConversationSessionRepository, StoredSession
from litellm.sessions.manager import client_session_key
from litellm.sessions.responses import provider_response_id

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()


class SessionItem(TypedDict):
    version: int
    seq: int
    is_output: bool
    summary: bool
    created_at: str
    content: object


class SessionResponse(TypedDict, total=False):
    id: str
    version: int
    item_count: int
    client_id: str


class SessionItemsQuery(BaseModel):
    response_ids: list[str] = Field(default_factory=list, max_length=1000)


class SessionItems(TypedDict):
    object: str
    session_id: str
    version: int
    items: list[SessionItem]
    responses: list[SessionResponse]


def _require_prisma() -> "PrismaClient":
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    return prisma_client


def _repository() -> ConversationSessionRepository:
    return ConversationSessionRepository(_require_prisma())


def _decoded(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else value


def _timestamp(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def stored_response_ids(advertised_ids: list[str], caller: UserAPIKeyAuth) -> dict[str, str]:
    """Map each advertised response id the caller may use to the provider id the store is keyed by"""
    security: Final = ResponsesIDSecurity()
    mapping: Final[dict[str, str]] = {}
    for advertised in advertised_ids:
        original = advertised
        if security._is_encrypted_response_id(advertised):  # pyright: ignore[reportPrivateUsage]  # same check as the hook
            original, user_id, team_id = security._decrypt_response_id(advertised)  # pyright: ignore[reportPrivateUsage]
            try:
                security.check_user_access_to_response_id(user_id, team_id, caller)
            except HTTPException:
                continue
        mapping[provider_response_id(original)] = advertised
    return mapping


_SUMMARY_KIND: Final = "compaction_summary"


def summary_ends(items: "list[object] | tuple[object, ...]") -> dict[int, int]:
    """For each version a compaction wrote, the seq of its summary item; items up to it are the summary"""
    ends: Final[dict[int, int]] = {}
    for item in items:
        meta = _decoded(getattr(item, "meta", None))
        if isinstance(meta, dict) and meta.get("kind") == _SUMMARY_KIND:
            version: int = getattr(item, "version")
            ends[version] = max(ends.get(version, -1), getattr(item, "seq"))
    return ends


def session_items(
    client_session_id: str, stored: StoredSession, advertised: dict[str, str] | None = None
) -> SessionItems:
    """The wire shape: items in version and transcript order, and each response with the prefix it completed"""
    items: Final = sorted(stored.items, key=lambda item: (item.version, item.seq))
    responses: Final = sorted(stored.responses, key=lambda response: (response.version, response.item_count))
    summaries: Final = summary_ends(items)
    return {
        "object": "session.items",
        "session_id": client_session_id,
        "version": stored.version,
        "items": [
            {
                "version": item.version,
                "seq": item.seq,
                "is_output": item.is_output,
                "summary": item.seq <= summaries.get(item.version, -1),
                "created_at": _timestamp(item.created_at),
                "content": _decoded(item.content),
            }
            for item in items
        ],
        "responses": [
            {
                "id": response.response_id,
                "version": response.version,
                "item_count": response.item_count,
                **(
                    {"client_id": advertised[response.response_id]}
                    if advertised and response.response_id in advertised
                    else {}
                ),
            }
            for response in responses
        ],
    }


@router.get(
    "/v1/sessions/{client_session_id}/items",
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_session_items(
    client_session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionItems:
    return session_items(client_session_id, await _load_owned(client_session_id, user_api_key_dict))


@router.post(
    "/v1/sessions/{client_session_id}/items",
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def resolve_session_items(
    client_session_id: str,
    query: SessionItemsQuery,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionItems:
    stored: Final = await _load_owned(client_session_id, user_api_key_dict)
    return session_items(client_session_id, stored, stored_response_ids(query.response_ids, user_api_key_dict))


async def _load_owned(client_session_id: str, user_api_key_dict: UserAPIKeyAuth) -> StoredSession:
    owner: Final = user_api_key_dict.api_key
    if not owner:
        raise HTTPException(status_code=401, detail="An API key is required to read a session")
    if litellm.session_manager is None:
        raise HTTPException(status_code=404, detail="The session store is not enabled on this gateway")
    # Turns are persisted by a background writer; drain it so a just-finished turn is readable.
    await litellm.session_manager.flush()
    stored: Final = await _repository().load_session_history(client_session_key(owner, client_session_id))
    if stored is None or stored.owner != owner:
        raise HTTPException(status_code=404, detail="Session not found")
    return stored
