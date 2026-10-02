from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, JsonValue

from litellm._logging import verbose_logger
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.sessions.compaction import turn_usage
from litellm.sessions.manager import Continuation, Forbidden, ReplayBase, Resolved, SessionManager
from litellm.sessions.provenance import Provenance
from litellm.sessions.session import Item

if TYPE_CHECKING:
    from litellm.types.llms.openai import ResponsesAPIResponse


def provider_response_id(response_id: str) -> str:
    decoded: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)  # pyright: ignore[reportPrivateUsage]  # sessions are keyed by the provider's own response id
    return str(decoded.get("response_id") or response_id)


_CACHE_KEYED_PROVIDERS: Final = frozenset({"openai", "azure", "xai"})
MANAGED_BY_CALLER: Final = "litellm_session_managed"
# How long a finished reply waits for its turn to reach the store before going out anyway
PERSIST_TIMEOUT_SECONDS: Final = 10.0


_NO_AFFINITY: Final[Mapping[str, str]] = MappingProxyType({})


def affinity_key(owner: str, client_session_id: str) -> str:
    """The upstream cache key for a client's session

    LiteLLM already derives `litellm_session_id` from the client's own headers, so the gateway does not invent a
    second identity for the cache: it keys on that one. Hashed with the owner because the value is client-chosen
    (two keys must not share a cache key, and the id itself must not travel upstream).
    """
    digest: Final = hashlib.sha256(f"{owner}\0{client_session_id}".encode()).hexdigest()
    return f"sess_{digest[:32]}"


def cache_affinity(custom_llm_provider: str | None, key: str, request: Mapping[str, object]) -> Mapping[str, str]:
    """Route every turn of a session to the same upstream prompt cache, unless the client chose a key itself"""
    if custom_llm_provider not in _CACHE_KEYED_PROVIDERS or request.get("prompt_cache_key") is not None:
        return _NO_AFFINITY
    return MappingProxyType({"prompt_cache_key": key})


def turn_affinity(turn: object, custom_llm_provider: str | None, request: Mapping[str, object]) -> Mapping[str, str]:
    return (
        cache_affinity(custom_llm_provider, turn.cache_key, request) if isinstance(turn, PendingTurn) else _NO_AFFINITY
    )


POLICY_METADATA_KEY: Final = "compaction_policy"


def requested_policy(*metadata: object) -> str | None:
    """The compaction policy a client named in its request metadata, first found wins"""
    for source in metadata:
        name = source.get(POLICY_METADATA_KEY) if isinstance(source, Mapping) else None  # rebind-ok: one per source
        if isinstance(name, str) and name:
            return name
    return None


def input_items(value: object) -> tuple[Item, ...]:
    if isinstance(value, str):
        return (
            {  # mutable-ok: JSON wire payload
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": value}],  # mutable-ok: JSON wire payload
            },
        )
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, dict))
    return ()


@dataclass(frozen=True, slots=True)
class PendingTurn:
    manager: SessionManager
    owner: str
    prior: Resolved | None
    turn_input: tuple[Item, ...]
    key: str
    started_at: float = 0.0
    replay: ReplayBase | None = None
    client_session_id: str | None = None
    compactable: bool = True
    provenance: Provenance | None = None

    @property
    def cache_key(self) -> str:
        """Stable across gateway restarts when the client names its session, so the upstream cache stays warm"""
        return affinity_key(self.owner, self.client_session_id) if self.client_session_id else self.key

    def replaying(self, replay: ReplayBase, record_only: bool = False) -> PendingTurn:
        return replace(self, replay=replay, compactable=self.compactable and not record_only)

    def upstream_input(self) -> list[Mapping[str, JsonValue]]:  # mutable-ok: provider request input is a JSON list
        history: Final = self.prior.history if self.prior is not None else ()
        return [*history, *self.turn_input]  # mutable-ok: provider request input is a JSON list

    def finish(self, response: ResponsesAPIResponse) -> None:
        if not response.id:
            return
        self.manager.commit(
            owner=self.owner,
            prior=self.prior,
            turn_input=self.turn_input,
            output=tuple(item for raw in response.output if (item := _as_item(raw)) is not None),
            response_id=provider_response_id(response.id),
            now=time.time(),
            new_session_id=self.key,
            started_at=self.started_at or None,
            usage=turn_usage(response.usage),
            replay=self.replay,
            client_session_id=self.client_session_id,
            compactable=self.compactable,
            provenance=self.provenance,
        )

    async def finish_durably(self, response: ResponsesAPIResponse, timeout: float = PERSIST_TIMEOUT_SECONDS) -> None:
        """`finish`, then hold the reply until the turn is stored

        The client may continue from this response id at any moment, and it may hold no other copy of the
        conversation. A gateway that answers first and stores later loses the turn when it is killed in between.
        A store that is down or slow does not fail the turn: the reply goes out and the miss is logged.
        """
        self.finish(response)
        if not response.id:
            return
        response_id: Final = provider_response_id(response.id)
        if not await self.manager.persisted(response_id, timeout):
            verbose_logger.error("SessionManager: turn %s was answered but is not stored", response_id)


def _as_item(item: object) -> Item | None:
    if isinstance(item, BaseModel):
        return item.model_dump(exclude_none=True)
    return item if isinstance(item, dict) else None


async def begin_turn(
    manager: SessionManager,
    previous_response_id: str | None,
    turn_input: object,
    owner: str,
    client_session_id: str | None = None,
    provenance: Provenance | None = None,
) -> PendingTurn | Forbidden | None:
    """None leaves the request to the existing continuation paths: the id predates the manager or was evicted"""
    items: Final = input_items(turn_input)
    started_at: Final = time.time()
    client: Final = client_session_id or None
    if not previous_response_id and client is None:
        return PendingTurn(manager, owner, None, items, uuid.uuid4().hex, started_at, provenance=provenance)
    if not previous_response_id:
        continuation: Final = await manager.continue_client_session(owner, client or "", items)
        return replace(_continuing(manager, owner, client, continuation, started_at), provenance=provenance)
    resolution: Final = await manager.resolve_async(provider_response_id(previous_response_id), owner)
    if isinstance(resolution, Resolved):
        manager.begin(resolution.session.id, started_at)
        return PendingTurn(
            manager,
            owner,
            resolution,
            paired_tool_outputs(resolution.history, items),
            resolution.session.id,
            started_at,
            client_session_id=client or resolution.session.client_session_id,
            provenance=provenance,
        )
    return resolution if isinstance(resolution, Forbidden) else None


_UNSAFE_CALL_ID_CHARS: Final = re.compile(r"[^A-Za-z0-9_-]")
_ALIAS_HASH_SUFFIX: Final = re.compile(r"_[0-9a-f]{10}$")
# Shortest clipped alias body still specific enough to name one call
_MIN_ALIAS_CORE: Final = 16


def _carries_call_id(alias: str, call_id: str) -> bool:
    """Whether a client's rewritten call id was derived from this one

    OpenClaw replays Responses tool ids in OpenAI's `call_*` shape: a call id such as Claude's `toolu_...` or
    xAI's `call-...-0` comes back as `call_<call_id|item_id, unsafe characters as _, clipped>_<10 hex>`.
    """
    safe: Final = _UNSAFE_CALL_ID_CHARS.sub("_", call_id)
    if safe in alias:
        return True
    core: Final = _ALIAS_HASH_SUFFIX.sub("", alias.removeprefix("call_"))
    return len(core) >= _MIN_ALIAS_CORE and _UNSAFE_CALL_ID_CHARS.sub("_", call_id.removeprefix("call_")).startswith(core)


def paired_tool_outputs(history: Sequence[Item], turn_input: tuple[Item, ...]) -> tuple[Item, ...]:
    """The turn's tool outputs, each under the call id the held history gave its call

    The gateway owns this history, so the provider sees the calls exactly as they were stored. A client that rewrote
    a call id in its own copy (OpenClaw normalizes ids to OpenAI's shape) sends its output under the rewritten id,
    and the provider then finds an unanswered call beside an orphaned output: OpenAI's "No tool output found for
    function call", claude_cli's "tool result IDs do not match pending Claude calls". An output whose id names no
    open call is moved to the one open call its id was derived from; anything ambiguous is left as sent.
    """
    answered: Final = {item.get("call_id") for item in history if item.get("type") == "function_call_output"}
    open_calls: Final = tuple(
        call_id
        for item in history
        if item.get("type") == "function_call"
        and isinstance(call_id := item.get("call_id"), str)
        and call_id not in answered
    )
    if not open_calls:
        return turn_input
    sent: Final = {item.get("call_id") for item in turn_input if item.get("type") == "function_call_output"}
    unclaimed: Final = [call_id for call_id in open_calls if call_id not in sent]  # mutable-ok: claimed one by one
    paired: Final[list[Item]] = []  # mutable-ok: built once, returned as a tuple
    for item in turn_input:
        alias = item.get("call_id")  # rebind-ok: per item
        owners = (  # rebind-ok: per item
            [call_id for call_id in unclaimed if _carries_call_id(alias, call_id)]
            if item.get("type") == "function_call_output" and isinstance(alias, str) and alias not in open_calls
            else []
        )
        if len(owners) != 1:
            paired.append(item)
            continue
        unclaimed.remove(owners[0])
        paired.append({**item, "call_id": owners[0]})
    return tuple(paired) if paired != list(turn_input) else turn_input


def _continuing(
    manager: SessionManager,
    owner: str,
    client: str | None,
    continuation: Continuation,
    started_at: float,
) -> PendingTurn:
    """The turn a client-named session resolved to, whether it continues one or starts it"""
    if continuation.prior is not None:
        manager.begin(continuation.session_id, started_at)
    return PendingTurn(
        manager,
        owner,
        continuation.prior,
        paired_tool_outputs(continuation.prior.history, continuation.turn_input)
        if continuation.prior is not None
        else continuation.turn_input,
        continuation.session_id,
        started_at,
        client_session_id=client,
        compactable=False,
    )


def client_session_id_of(*sources: object) -> str | None:
    """The session id LiteLLM derived for this request (headers, body or baggage), first found wins"""
    for source in sources:
        value = source.get("litellm_session_id") if isinstance(source, Mapping) else source  # rebind-ok: per source
        if isinstance(value, str) and value:
            return value
    return None
