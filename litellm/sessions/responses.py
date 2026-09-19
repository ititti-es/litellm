from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, JsonValue

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.sessions.manager import Forbidden, Resolved, SessionManager
from litellm.sessions.session import Item

if TYPE_CHECKING:
    from litellm.types.llms.openai import ResponsesAPIResponse


def provider_response_id(response_id: str) -> str:
    decoded: Final = ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)  # pyright: ignore[reportPrivateUsage]  # sessions are keyed by the provider's own response id
    return str(decoded.get("response_id") or response_id)


_CACHE_KEYED_PROVIDERS: Final = frozenset({"openai", "azure", "xai"})
MANAGED_BY_CALLER: Final = "litellm_session_managed"


_NO_AFFINITY: Final[Mapping[str, str]] = MappingProxyType({})


def cache_affinity(custom_llm_provider: str | None, key: str, request: Mapping[str, object]) -> Mapping[str, str]:
    """Route every turn of a session to the same upstream prompt cache, unless the client chose a key itself"""
    if custom_llm_provider not in _CACHE_KEYED_PROVIDERS or request.get("prompt_cache_key") is not None:
        return _NO_AFFINITY
    return MappingProxyType({"prompt_cache_key": key})


def turn_affinity(turn: object, custom_llm_provider: str | None, request: Mapping[str, object]) -> Mapping[str, str]:
    return cache_affinity(custom_llm_provider, turn.key, request) if isinstance(turn, PendingTurn) else _NO_AFFINITY


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
        )


def _as_item(item: object) -> Item | None:
    if isinstance(item, BaseModel):
        return item.model_dump(exclude_none=True)
    return item if isinstance(item, dict) else None


def begin_turn(
    manager: SessionManager, previous_response_id: str | None, turn_input: object, owner: str
) -> PendingTurn | Forbidden | None:
    """None leaves the request to the existing continuation paths: the id predates the manager or was evicted"""
    items: Final = input_items(turn_input)
    if not previous_response_id:
        return PendingTurn(manager, owner, None, items, uuid.uuid4().hex)
    resolution: Final = manager.resolve(provider_response_id(previous_response_id), owner)
    if isinstance(resolution, Resolved):
        return PendingTurn(manager, owner, resolution, items, resolution.session.id)
    return resolution if isinstance(resolution, Forbidden) else None
