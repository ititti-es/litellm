from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, TypeAlias, cast

from pydantic import JsonValue

Item: TypeAlias = Mapping[str, JsonValue]
Meta: TypeAlias = Mapping[str, JsonValue]


def _no_annotations() -> Mapping[int, Meta]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Session:
    id: str
    owner: str
    items: tuple[Item, ...]
    outputs: frozenset[int]
    responses: Mapping[str, int]
    version: int
    last_upstream_at: float
    annotations: Mapping[int, Meta] = field(default_factory=_no_annotations)
    """Item meta by position (e.g. what a compaction summary covers); stored beside the item, never replayed"""
    client_session_id: str | None = None
    """LiteLLM's `litellm_session_id` for this conversation, so the store shares the client's own identity"""
    superseded: bool = False
    """An earlier version a compaction replaced, loaded so its response ids still resolve; never extended in place"""

    def replayable_at(self, response_id: str) -> tuple[Item, ...]:
        """History up to *response_id*, with model output reduced to what a stateless upstream accepts back"""
        end: Final = self.responses[response_id]
        return tuple(
            projected
            for index, item in enumerate(self.items[:end])
            if (projected := _replay_output(item) if index in self.outputs else item) is not None
        )

    def size(self) -> int:
        return len(canonical(self.items))

    def state_hash(self) -> str:
        return "sha256:" + hashlib.sha256(canonical(self.items)).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


Form: TypeAlias = tuple[str, ...]


def comparable(items: Sequence[Item]) -> tuple[tuple[int, Form], ...]:
    """Each item's position and what it says, so two renderings of one history can be compared.

    A client that keeps its own history resends the whole conversation every turn, and its rendering is not
    byte-identical to what the gateway stored: ids, annotations, status fields and the exact shape of a
    message all differ between a harness and a provider. What has to match is what was said, and in what
    order. Reasoning is left out of the comparison entirely -- it is dropped on replay unless it is sealed,
    and a harness is free not to echo it back.
    """
    return tuple((index, form) for index, item in enumerate(items) if (form := _form(item)) is not None)


# Postgres refuses a tsvector over 1 MiB, and the full-text index covers this column, so one huge tool output
# would fail the whole write. The cap is far below that limit even for text made of distinct words.
SEARCH_TEXT_LIMIT: Final = 200_000


def search_text(item: Item) -> str | None:
    kind: Final = item.get("type", "message")
    if kind == "reasoning":
        return None
    text: Final = (
        _string(item, "arguments")
        if kind == "function_call"
        else _searchable_text(item.get("output") if kind == "function_call_output" else item.get("content"))
    )
    return text[:SEARCH_TEXT_LIMIT] or None


def _searchable_text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(text for part in content if (text := _searchable_part(part)))


def _searchable_part(part: JsonValue) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict) or part.get("type") not in (None, "text", "input_text", "output_text"):
        return ""
    text: Final = part.get("text")
    return text if isinstance(text, str) else ""


def _form(item: Item) -> Form | None:
    kind: Final = item.get("type", "message")
    if kind == "reasoning":
        return None
    if kind == "function_call":
        return (
            "function_call",
            _string(item, "call_id"),
            _string(item, "name"),
            _arguments(_string(item, "arguments")),
        )
    if kind == "function_call_output":
        return ("function_call_output", _string(item, "call_id"), _text_of(item.get("output")))
    role: Final = _string(item, "role")
    text: Final = _text_of(item.get("content"))
    if role == "assistant" and not text:
        return None
    return ("message", role, text)


def _arguments(arguments: str) -> str:
    """A tool call's arguments as what they ask for rather than how they were encoded

    A harness that parses a call and re-encodes it resends the same request in other bytes: 50.0 comes back as
    50, keys in another order, an argument string that was itself JSON-encoded comes back as the object. Text
    that is not JSON is compared as it is.
    """
    try:
        decoded = _decoded_json(arguments)  # rebind-ok: unwrapped once more if it was a JSON string
        if isinstance(decoded, str):
            decoded = _decoded_json(decoded)
    except ValueError:
        return arguments
    return json.dumps(_integral(decoded), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _decoded_json(text: str) -> JsonValue:
    return cast(JsonValue, json.loads(text))


def _integral(value: JsonValue) -> JsonValue:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_integral(entry) for entry in value]  # mutable-ok: rebuilt JSON, only serialized
    if isinstance(value, dict):
        return {key: _integral(entry) for key, entry in value.items()}  # mutable-ok: same
    return value


def _string(item: Item, key: str) -> str:
    value: Final = item.get(key)
    return value if isinstance(value, str) else ""


def _text_of(content: JsonValue) -> str:
    """What a message says, however the harness wrapped it; a part carrying no text is named, not ignored."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(_part_text(part) for part in content)


def _part_text(part: JsonValue) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    text: Final = part.get("text")
    return text if isinstance(text, str) else f"[{part.get('type')}]"


_CLIENT_CALLS: Final = frozenset(
    {"function_call", "custom_tool_call", "local_shell_call", "computer_call", "shell_call", "apply_patch_call"}
)
# An `fc_`/`ctc_` id makes OpenAI demand the reasoning item it followed, which is dropped unless sealed
_ID_OPTIONAL_CALLS: Final = frozenset({"function_call", "custom_tool_call"})


def _replay_output(item: Item) -> Item | None:
    kind: Final = item.get("type")
    if kind in _ID_OPTIONAL_CALLS:
        return {key: value for key, value in item.items() if key != "id"}  # mutable-ok: JSON wire payload
    if kind in _CLIENT_CALLS:
        return item
    if kind == "reasoning":
        return item if item.get("encrypted_content") else None
    if kind != "message":
        return None
    content: Final = item.get("content")
    text: Final = "".join(
        str(part.get("text", ""))
        for part in (content if isinstance(content, list) else ())
        if isinstance(part, dict) and part.get("type") in ("output_text", "text")
    )
    if not text:
        return None
    return {  # mutable-ok: replayed items go to litellm.aresponses as JSON input
        "type": "message",
        "role": item.get("role", "assistant"),
        "content": [{"type": "output_text", "text": text}],  # mutable-ok: same JSON wire payload
    }
