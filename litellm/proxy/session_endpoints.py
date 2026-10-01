"""Read access to the gateway-owned session store.

GET  /v1/sessions/{client_session_id}/items
POST /v1/sessions/{client_session_id}/items  {"response_ids": [...]}

Returns the current transcript of a conversation a client named with its session id (the
``litellm_session_id`` taken from ``x-litellm-session-id``, ``session_id`` and the other recognised
headers), so a client that delegates history to this gateway can show it without keeping its own copy.

The stored row is found the same way a turn finds it, by ``client_session_key(owner, client_session_id)``,
and the owner is the caller's hashed API key. A key therefore only ever reads conversations it wrote: any
other key derives a different row id, and a row whose owner does not match is reported as missing.

The store keys responses by the provider's own id, while a client only ever saw the id the gateway advertised
(possibly encrypted by the response-id security hook). POST names the advertised ids the client holds, and
each stored response it can be matched to carries that id back as ``client_id``, so the client can join its
own records to the stored turns without decoding gateway ids itself.
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
    seq: int
    is_output: bool
    created_at: str
    content: object


class SessionResponse(TypedDict, total=False):
    id: str
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


def session_items(
    client_session_id: str, stored: StoredSession, advertised: dict[str, str] | None = None
) -> SessionItems:
    """The wire shape: items in transcript order, and each response id with the prefix length it completed"""
    items: Final = sorted(stored.items, key=lambda item: item.seq)
    responses: Final = sorted(stored.responses, key=lambda response: response.item_count)
    return {
        "object": "session.items",
        "session_id": client_session_id,
        "version": stored.version,
        "items": [
            {
                "seq": item.seq,
                "is_output": item.is_output,
                "created_at": _timestamp(item.created_at),
                "content": _decoded(item.content),
            }
            for item in items
        ],
        "responses": [
            {
                "id": response.response_id,
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
    stored: Final = await _repository().load_session(client_session_key(owner, client_session_id))
    if stored is None or stored.owner != owner:
        raise HTTPException(status_code=404, detail="Session not found")
    return stored
