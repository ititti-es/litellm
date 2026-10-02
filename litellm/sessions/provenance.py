"""Which agent and which harness a conversation came from, so stored sessions can be filtered by them.

The agent is the named assistant (an OpenClaw agent such as "sett"); the harness is the client program that spoke
for it (openclaw, opencode, claude_code, codex). Both are read from the request that commits a turn and kept on the
session, so a later turn that does not say carries the earlier value forward.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter, ValidationError

AGENT_HEADERS: Final = ("x-litellm-agent", "x-openclaw-agent-id", "x-openclaw-agent")
HARNESS_HEADER: Final = "x-litellm-harness"
_MAX_LENGTH: Final = 64
_OBJECT: Final = TypeAdapter(dict[str, object])
_NOTHING: Final[Mapping[str, object]] = MappingProxyType({})
_TOKEN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_KNOWN_HARNESSES: Final = (
    ("openclaw", "openclaw"),
    ("opencode", "opencode"),
    ("claude-cli", "claude_code"),
    ("claude-code", "claude_code"),
    ("codex", "codex"),
)


@dataclass(frozen=True, slots=True)
class Provenance:
    agent: str | None = None
    harness: str | None = None

    def over(self, earlier: Provenance | None) -> Provenance | None:
        """This turn's provenance, with whatever it leaves unsaid taken from the session's earlier turns"""
        if earlier is None:
            return self if self.agent or self.harness else None
        return Provenance(agent=self.agent or earlier.agent, harness=self.harness or earlier.harness)


def _label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match: Final = _TOKEN.match(value.strip())
    return match.group(0)[:_MAX_LENGTH].lower() if match else None


def harness_of(user_agent: object) -> str | None:
    """A short harness name from a User-Agent: a known client by name, otherwise the first product token"""
    if not isinstance(user_agent, str) or not user_agent.strip():
        return None
    lowered: Final = user_agent.lower()
    known: Final = next((name for marker, name in _KNOWN_HARNESSES if lowered.startswith(marker)), None)
    return known or _label(user_agent.split("/", 1)[0])


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return MappingProxyType(_OBJECT.validate_python(value)) if value is not None else _NOTHING
    except ValidationError:
        return _NOTHING


def _header(headers: Mapping[str, object], names: tuple[str, ...]) -> str | None:
    lowered: Final = MappingProxyType({key.lower(): value for key, value in headers.items()})
    return next((label for name in names if (label := _label(lowered.get(name))) is not None), None)


def provenance_of(metadata: object) -> Provenance | None:
    """Agent and harness for a request, from LiteLLM's request metadata: explicit headers first, then the
    calling key's metadata, then (for the harness) the User-Agent"""
    request: Final = _mapping(metadata)
    if not request:
        return None
    headers: Final = _mapping(request.get("headers"))
    key: Final = _mapping(request.get("user_api_key_metadata"))
    agent: Final = _header(headers, AGENT_HEADERS) or _label(key.get("agent"))
    harness: Final = (
        _header(headers, (HARNESS_HEADER,)) or _label(key.get("harness")) or harness_of(request.get("user_agent"))
    )
    return Provenance(agent=agent, harness=harness).over(None)


def provenance_from_stored(metadata: object) -> Provenance | None:
    stored: Final = _mapping(metadata)
    return Provenance(agent=_label(stored.get("agent")), harness=_label(stored.get("harness"))).over(None)


def stored_fields(provenance: Provenance | None) -> Mapping[str, str]:
    if provenance is None:
        return MappingProxyType({})
    return MappingProxyType(
        {
            name: value
            for name, value in (("agent", provenance.agent), ("harness", provenance.harness))
            if value is not None
        }
    )
