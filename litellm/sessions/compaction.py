"""Compaction while the upstream cache is still warm

Just before a session's cached prefix is predicted to expire, the gateway sends the provider exactly the prefix
the next turn would send (same model, instructions, tools and cache key), plus one instruction to summarise.
The long history is read at cached rates; the session is then rebased onto the summary and its latest
exchange, so the turn after the cache goes cold starts from a short prefix instead of the full one.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, JsonValue

from litellm._logging import verbose_logger
from litellm.sessions.manager import CompactionJob, ReplayBase, SessionManager, TurnUsage
from litellm.sessions.session import Item, Meta, Session

DEFAULT_POLICY: Final = "default"
DEFAULT_INSTRUCTIONS: Final = """\
Stop working on the task for a moment. The conversation above is about to be compacted: everything before the
most recent user message will be replaced by the summary you write now, and you will continue from that summary
alone. Do not call any tools.

Write a summary that lets you pick the work up without the original transcript:
- the user's goal and every explicit requirement, preference or constraint they stated;
- decisions made, and why;
- the current state: files, commands, identifiers, URLs and values that matter, quoted exactly;
- what was tried and failed, so it is not repeated;
- open questions and the next steps.

Be specific and complete rather than short. Write it as notes to yourself, without preamble."""
SUMMARY_PREFIX: Final = "Summary of the conversation so far (earlier turns were compacted):\n\n"

# Parameters that shape the cached prefix upstream; anything else (stream, output limits) is left out
REPLAY_PARAMS: Final = frozenset(
    {
        "instructions",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "text",
        "truncation",
        "temperature",
        "top_p",
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "include",
        "store",
        "extra_headers",
        "extra_body",
    }
)


def replay_params(request: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {name: value for name, value in request.items() if name in REPLAY_PARAMS and value is not None}
    )


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    name: str
    instructions: str


DEFAULT: Final = CompactionPolicy(DEFAULT_POLICY, DEFAULT_INSTRUCTIONS)


@dataclass(frozen=True, slots=True)
class CompactionReply:
    text: str
    called_tools: bool
    usage: TurnUsage | None


class CallUpstream(Protocol):
    """Sends the input upstream, without streaming, as the replayed turn would have been; returns the reply"""

    def __call__(self, replay: ReplayBase, items: Sequence[Item], owner: str) -> Awaitable[CompactionReply]: ...


def instruction_item(policy: CompactionPolicy) -> Item:
    return {  # mutable-ok: sent upstream as JSON input
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": policy.instructions}],  # mutable-ok: same JSON payload
    }


def summary_item(text: str) -> Item:
    return {  # mutable-ok: replayed upstream as JSON input
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": SUMMARY_PREFIX + text}],  # mutable-ok: same JSON payload
    }


def summary_meta(job: CompactionJob, policy: CompactionPolicy, reply: CompactionReply, started_at: float) -> Meta:
    """The summary's provenance: the span it replaced, and what it cost to write (the seed of a memory node)"""
    usage: Final[Mapping[str, JsonValue]] = (
        MappingProxyType(
            {
                "input_tokens": reply.usage.input_tokens,
                "cached_tokens": reply.usage.cached_tokens,
                "output_tokens": reply.usage.output_tokens,
            }
        )
        if reply.usage is not None
        else MappingProxyType({})
    )
    return MappingProxyType(
        {
            "kind": "compaction_summary",
            "summary": reply.text,
            "policy": policy.name,
            "provenance": MappingProxyType(
                {"harness": "litellm", "provider": job.replay.provider, "model": job.replay.model}
            ),
            "covers": MappingProxyType(
                {"version": job.session.version, "first_seq": 0, "last_seq": job.tail_start - 1}
            ),
            "usage": usage,
            "started_at": started_at,
            "warm_until": job.warm_until,
        }
    )


async def compact(
    manager: SessionManager, job: CompactionJob, policy: CompactionPolicy, call: CallUpstream
) -> Session | None:
    """One compaction; None when it was not usable or the session moved on while it ran"""
    started_at: Final = time.time()
    try:
        request: Final = [*job.history, instruction_item(policy)]  # mutable-ok: provider input is a JSON list
        reply: Final = await call(job.replay, request, job.session.owner)
    except Exception as exc:
        verbose_logger.warning("SessionManager: compaction of %s failed: %s", job.session.id, exc)
        return None
    if reply.called_tools or not reply.text.strip():
        verbose_logger.warning("SessionManager: compaction of %s returned no usable summary", job.session.id)
        return None
    return manager.apply_compaction(
        job, summary_item(reply.text.strip()), summary_meta(job, policy, reply, started_at), time.time()
    )


def reply_from(response: object) -> CompactionReply:
    """The text, tool use and usage of a non-streamed Responses API reply"""
    output: Final = _field(response, "output")
    items: Final = tuple(_plain(item) for item in (output if isinstance(output, list) else ()))
    text: Final = "".join(
        str(part.get("text", ""))
        for item in items
        if item.get("type") == "message"
        for part in _parts(item.get("content"))
        if part.get("type") in ("output_text", "text")
    )
    return CompactionReply(
        text=text,
        called_tools=any(item.get("type") == "function_call" for item in items),
        usage=turn_usage(_field(response, "usage")),
    )


def turn_usage(usage: object) -> TurnUsage | None:
    """Responses API usage, from a model or its JSON form"""
    plain: Final = _plain(usage)
    input_tokens: Final = plain.get("input_tokens")
    if not isinstance(input_tokens, int):
        return None
    details: Final = _plain(plain.get("input_tokens_details"))
    cached: Final = details.get("cached_tokens")
    output_tokens: Final = plain.get("output_tokens")
    return TurnUsage(
        input_tokens=input_tokens,
        cached_tokens=cached if isinstance(cached, int) else 0,
        output_tokens=output_tokens if isinstance(output_tokens, int) else 0,
    )


def _field(value: object, name: str) -> object:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _plain(value: object) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return value if isinstance(value, Mapping) else MappingProxyType({})


def _parts(content: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_plain(part) for part in content) if isinstance(content, list | tuple) else ()
