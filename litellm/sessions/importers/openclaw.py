"""openclaw sessions (per-agent `openclaw-agent.sqlite`) as harness-neutral transcripts

Each transcript generation (`session_windows` row) becomes one session: its entries form a tree, and the
imported history is the active path from the recorded leaf back to the root
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from itertools import accumulate
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from litellm.sessions.importers.common import (
    ImportedItem,
    ImportedSession,
    Meta,
    assistant_text,
    function_call,
    function_call_output,
    input_image,
    input_text,
    reasoning,
    sqlite_snapshot,
    user_message,
)
from litellm.sessions.session import Item

SOURCE: Final = "openclaw"
_SUMMARY_KINDS: Final = MappingProxyType(
    {
        "compaction": "compaction_summary",
        "compactionSummary": "compaction_summary",
        "branch_summary": "branch_summary",
        "branchSummary": "branch_summary",
    }
)
_IS_ERROR: Final[Meta] = MappingProxyType({"is_error": True})
_NO_ERROR: Final[Meta] = MappingProxyType({})
_CONTROL: Final = frozenset(
    {"session", "custom", "model_change", "thinking_level_change", "label", "session_info", "leaf", "reset"}
)


def read_openclaw(state_dir: Path, scratch_dir: Path | None = None) -> Iterator[ImportedSession]:
    for db_path in sorted(state_dir.glob("agents/*/agent/openclaw-agent.sqlite")):
        agent_id: str = db_path.parent.parent.name  # rebind-ok: one agent per database
        with sqlite_snapshot(db_path, scratch_dir) as db:
            yield from _agent_sessions(db, agent_id)


def _agent_sessions(db: sqlite3.Connection, agent_id: str) -> Iterator[ImportedSession]:
    leaves: Final = MappingProxyType(
        {
            str(sid): str(leaf)
            for sid, leaf in db.execute("select session_id, leaf_event_id from session_transcript_index_state")
            if leaf
        }
    )
    current: Final = MappingProxyType(
        {
            str(key): str(sid)
            for key, sid in db.execute("select session_key, current_session_id from session_nodes")
            if sid
        }
    )
    windows: Final = db.execute(
        "select session_id, session_key, previous_session_id, reason, created_at, updated_at, channel, "
        "parent_session_key, display_name from session_windows order by created_at, session_id"
    ).fetchall()
    for window in windows:
        yield _window(db, agent_id, window, leaves, current)


def _window(
    db: sqlite3.Connection,
    agent_id: str,
    window: tuple[object, ...],
    leaves: Mapping[str, str],
    current: Mapping[str, str],
) -> ImportedSession:
    session_id, key, previous, reason, created, updated, channel, parent_key, display_name = window
    entries: Final = tuple(
        _object(json.loads(str(raw)))
        for (raw,) in db.execute(
            "select event_json from transcript_events where session_id = ? order by seq", (session_id,)
        )
    )
    header: Final = next((entry for entry in entries if entry.get("type") == "session"), MappingProxyType({}))
    path: Final = _active_path(entries, leaves.get(str(session_id)))
    per_entry: Final = tuple(_entry_items(entry, agent_id, str(header.get("version", ""))) for entry in path)
    ends: Final = tuple(accumulate(len(items) for items, _ in per_entry))
    parent: Final = current.get(str(parent_key)) if parent_key else None
    return ImportedSession(
        source=SOURCE,
        source_id=str(session_id),
        title=str(display_name) if display_name else None,
        parent_source_id=parent,
        created_at=_ms(created),
        updated_at=_ms(updated),
        items=tuple(item for items, _ in per_entry for item in items),
        turn_ends=tuple(end for end, (items, is_turn) in zip(ends, per_entry) if is_turn and items),
        metadata=MappingProxyType(
            {
                "harness": SOURCE,
                "harness_version": str(header.get("version", "")),
                "agent": agent_id,
                "session_key": str(key),
                "window_reason": str(reason or ""),
                "channel": str(channel or ""),
                "previous_source_id": str(previous) if previous else None,
                "cwd": str(header.get("cwd", "")),
            }
        ),
        skipped=sum(1 for entry in path if _unrecognized(entry)),
    )


def _active_path(entries: Sequence[Mapping[str, JsonValue]], leaf: str | None) -> tuple[Mapping[str, JsonValue], ...]:
    by_id: Final = MappingProxyType({str(entry["id"]): entry for entry in entries if entry.get("id") is not None})
    start: Final = (
        leaf
        if leaf in by_id
        else next(
            (
                str(entry["id"])
                for entry in reversed(entries)
                if entry.get("id") is not None and entry.get("type") != "leaf"
            ),
            None,
        )
    )
    return tuple(reversed(tuple(_ancestors(by_id, start))))


def _ancestors(by_id: Mapping[str, Mapping[str, JsonValue]], start: str | None) -> Iterator[Mapping[str, JsonValue]]:
    seen: Final[set[str]] = set()  # mutable-ok: cycle guard while walking parent links
    node: str | None = start  # rebind-ok: walks from the leaf to the root
    while node is not None and node in by_id and node not in seen:
        seen.add(node)
        entry: Mapping[str, JsonValue] = by_id[node]  # rebind-ok: one entry per step
        yield entry
        parent_id: JsonValue = entry.get("parentId")  # rebind-ok: one link per step
        node = str(parent_id) if parent_id is not None else None


def _entry_items(entry: Mapping[str, JsonValue], agent_id: str, version: str) -> tuple[tuple[ImportedItem, ...], bool]:
    """The entry's items and whether it is an assistant turn"""
    kind: Final = str(entry.get("type", ""))
    source: Final = MappingProxyType({"entry_id": entry.get("id"), "entry": entry})
    if kind in _SUMMARY_KINDS:
        return (_note(str(entry.get("summary", "")), _SUMMARY_KINDS[kind], source),), False
    if kind == "custom_message":
        return (_note(_text(entry.get("content")), "harness_context", source),), False
    if kind != "message":
        return (), False
    message: Final = _object(entry.get("message"))
    role: Final = message.get("role")
    if role == "user":
        content: Final = _user_content(message.get("content"))
        return (
            (user_message(content, MappingProxyType({"kind": "user", "source": source})),) if content else ()
        ), False
    if role == "assistant":
        return _assistant_items(message, source, agent_id, version), True
    if role == "toolResult":
        return (_tool_result(message, source),), False
    if role in _SUMMARY_KINDS:
        return (_note(str(message.get("summary", "")), _SUMMARY_KINDS[str(role)], source),), False
    if role == "bashExecution":
        return (_note(f"Ran `{message.get('command', '')}`\n{message.get('output', '')}", "shell_note", source),), False
    if role == "custom" and not message.get("excludeFromContext"):
        return (_note(_text(message.get("content")), "harness_context", source),), False
    return (), False


def _unrecognized(entry: Mapping[str, JsonValue]) -> bool:
    kind: Final = str(entry.get("type", ""))
    return kind not in _CONTROL and kind not in _SUMMARY_KINDS and kind not in ("message", "custom_message")


def _note(text: str, kind: str, source: Meta) -> ImportedItem:
    return user_message((input_text(text),), MappingProxyType({"kind": kind, "source": source}))


def _assistant_items(
    message: Mapping[str, JsonValue], source: Meta, agent_id: str, version: str
) -> tuple[ImportedItem, ...]:
    provenance: Final = MappingProxyType(
        {
            "harness": SOURCE,
            "harness_version": version,
            "provider": message.get("provider"),
            "model": message.get("responseModel") or message.get("model"),
            "api": message.get("api"),
            "agent": agent_id,
            "stop_reason": message.get("stopReason"),
        }
    )
    meta: Final = MappingProxyType({"source": source, "provenance": provenance})
    blocks: Final = message.get("content")
    return tuple(
        item
        for block in (blocks if isinstance(blocks, list) else ())
        if isinstance(block, dict) and (item := _assistant_block(block, meta, provenance)) is not None
    )


def _assistant_block(block: Mapping[str, JsonValue], meta: Meta, provenance: Meta) -> ImportedItem | None:
    kind: Final = block.get("type")
    if kind == "text" and block.get("text"):
        return assistant_text(str(block["text"]), MappingProxyType({**meta, "kind": "text"}))
    if kind == "thinking":
        signed: Final = _signed_reasoning(block.get("thinkingSignature"))
        encrypted: Final = signed.get("encrypted_content") if isinstance(signed.get("encrypted_content"), str) else None
        bound: Final = (
            MappingProxyType({**meta, "kind": "reasoning", "provider": provenance.get("provider")})
            if encrypted
            else MappingProxyType({**meta, "kind": "reasoning"})
        )
        return reasoning(str(block.get("thinking", "")), str(encrypted) if encrypted else None, bound)
    if kind == "toolCall":
        return function_call(
            str(block.get("id", "")).split("|")[0],
            str(block.get("name", "")),
            block.get("arguments") or "{}",
            MappingProxyType({**meta, "kind": "tool_call"}),
        )
    return None


def _signed_reasoning(signature: JsonValue) -> Mapping[str, JsonValue]:
    """openclaw keeps an OpenAI reasoning item, encrypted content included, as the thinking signature"""
    if not isinstance(signature, str):
        return MappingProxyType({})
    try:
        parsed: Final = json.loads(signature)
    except ValueError:
        return MappingProxyType({})
    return parsed if isinstance(parsed, dict) and parsed.get("type") == "reasoning" else MappingProxyType({})


def _tool_result(message: Mapping[str, JsonValue], source: Meta) -> ImportedItem:
    error: Final = _IS_ERROR if message.get("isError") else _NO_ERROR
    return function_call_output(
        str(message.get("toolCallId", "")).split("|")[0],
        _text(message.get("content")),
        MappingProxyType({"kind": "tool_output", "source": source, **error}),
    )


def _user_content(content: JsonValue) -> tuple[Item, ...]:
    if isinstance(content, str):
        return (input_text(content),) if content else ()
    return tuple(
        block for raw in (content if isinstance(content, list) else ()) if (block := _user_block(raw)) is not None
    )


def _user_block(block: JsonValue) -> Item | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "text" and block.get("text"):
        return input_text(str(block["text"]))
    if block.get("type") == "image" and block.get("data"):
        return input_image(f"data:{block.get('mimeType', 'image/png')};base64,{block['data']}")
    return None


def _text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    blocks: Final = content if isinstance(content, list) else []  # mutable-ok: read-only view of a JSON list
    return "\n".join(
        str(block.get("text", "")) if block.get("type") == "text" else f"[{block.get('type')}]"
        for block in blocks
        if isinstance(block, dict)
    )


def _object(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, dict) else MappingProxyType({})


def _ms(value: object) -> datetime:
    return datetime.fromtimestamp((value if isinstance(value, int | float) else 0) / 1000, UTC)
