from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, TypeAlias

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


def _replay_output(item: Item) -> Item | None:
    kind: Final = item.get("type")
    if kind == "function_call":
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
