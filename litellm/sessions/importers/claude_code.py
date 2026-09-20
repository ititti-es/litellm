"""Claude Code sessions (`~/.claude/projects/<project>/<session-id>.jsonl`) as harness-neutral transcripts

Each file is one session, appended a record at a time. The records carry `uuid`/`parentUuid`, so a session
that was branched or compacted holds more than one history; the imported one is the active path back from
the last record written. Compaction cuts that chain -- the boundary record's `parentUuid` is null and the
history it replaced hangs off `logicalParentUuid` -- so the walk follows the logical link too, keeping the
turns the harness dropped and the summary that replaced them, each marked for a view to decide about.

This is also how the `claude_cli` provider's transcripts reach the store: Claude Code owns that conversation
and only its last exchange passes through the proxy, so the record has to come from here.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
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
    user_message,
)
from litellm.sessions.session import Item

SOURCE: Final = "claude_code"
DEFAULT_PROJECTS_DIR: Final = Path.home() / ".claude" / "projects"

# Records that are the harness talking to itself about the turn it just ran, not the conversation.
_CONTROL_SUBTYPES: Final = frozenset({"turn_duration", "informational", "model_refusal_fallback"})

# Attachments the harness rebuilds from live state on every turn. They are context, but re-importing them
# would store thousands of copies of a token counter, so only the durable ones are kept.
_EPHEMERAL_ATTACHMENTS: Final = frozenset(
    {
        "total_tokens_reminder",
        "batching_reminder_sent",
        "silent_turn_reminder",
        "task_reminder",
        "prompt_snapshot",
        "queued_command",
        "thinking_drop",
    }
)

_TOOL_OUTPUT: Final[Meta] = MappingProxyType({"kind": "tool_output"})
_ERROR_OUTPUT: Final[Meta] = MappingProxyType({"kind": "tool_output", "is_error": True})
_NO_ITEMS: Final[tuple[ImportedItem, ...]] = ()
_EMPTY: Final[Mapping[str, JsonValue]] = MappingProxyType({})


def transcript_path(session_id: str, projects_dir: Path | None = None) -> Path | None:
    """Where Claude Code kept `session_id`, if it kept it anywhere.

    The directory under `projects/` is named after the working directory the session ran in, which a caller
    that only started the session through `--session-id` has no reason to know, so it is searched for.
    """
    root: Final = projects_dir or DEFAULT_PROJECTS_DIR
    direct: Final = sorted(root.glob(f"*/{session_id}.jsonl"))
    return direct[0] if direct else None


def read_claude_code(root: Path, scratch_dir: Path | None = None) -> Iterator[ImportedSession]:
    """Every session under a `projects/` directory, or the single transcript `root` names."""
    for path in (root,) if root.is_file() else sorted(root.glob("*/*.jsonl")):
        session: ImportedSession | None = _session(path, scratch_dir)  # rebind-ok: one session per transcript
        if session is not None:
            yield session


def read_claude_code_session(
    session_id: str, projects_dir: Path | None = None, scratch_dir: Path | None = None
) -> ImportedSession | None:
    """One session by id, for syncing a conversation the proxy has just added a turn to."""
    path: Final = transcript_path(session_id, projects_dir)
    return _session(path, scratch_dir) if path is not None else None


@contextmanager
def _readable(path: Path, scratch_dir: Path | None) -> Generator[Path]:
    """The transcript, or a private copy of it.

    A transcript is only ever appended to, so reading the live file is safe: the worst a concurrent write
    can do is leave the final line half-written, which parses as nothing and is counted as skipped. A copy
    is offered anyway for a caller that wants the harness's file untouched by even a read.
    """
    if scratch_dir is None:
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="litellm-import-", dir=scratch_dir) as scratch:
        target: Final = Path(scratch) / path.name
        shutil.copy2(path, target)
        yield target


def _records(path: Path) -> tuple[tuple[Mapping[str, JsonValue], ...], int]:
    """The transcript's records, and how many lines were not readable as one."""
    parsed: list[Mapping[str, JsonValue]] = []  # mutable-ok: accumulator, returned as a tuple
    unreadable = 0  # rebind-ok: counts malformed lines
    with open(path, encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)  # rebind-ok: one record per line
            except ValueError:
                unreadable += 1
                continue
            if isinstance(record, dict):
                parsed.append(record)
            else:
                unreadable += 1
    return tuple(parsed), unreadable


def _session(path: Path, scratch_dir: Path | None) -> ImportedSession | None:
    with _readable(path, scratch_dir) as readable:
        records, unreadable = _records(readable)
    if not records:
        return None
    history: Final = _active_path(records)
    if not history:
        return None
    version: Final = _last_string(records, "version")
    converted: Final = tuple(_record_items(record, version) for record in history)
    ends: Final = tuple(accumulate(len(items) for items, _ in converted))
    sidechains: Final = sum(1 for record in records if record.get("isSidechain"))
    return ImportedSession(
        source=SOURCE,
        source_id=path.stem,
        title=_title(records),
        parent_source_id=None,
        created_at=_at(history[0]),
        updated_at=_at(history[-1]),
        items=tuple(item for items, _ in converted for item in items),
        turn_ends=_turn_ends(history, converted, ends),
        metadata=MappingProxyType(
            {
                "harness": SOURCE,
                "harness_version": version,
                "cwd": _last_string(records, "cwd"),
                "git_branch": _last_string(records, "gitBranch"),
                "entrypoint": _last_string(records, "entrypoint"),
                "model": _last_model(records),
                "project": path.parent.name,
                "sidechain_records": sidechains,
            }
        ),
        skipped=unreadable + sidechains + sum(1 for record in history if _unrecognized(record)),
    )


def _turn_ends(
    history: Sequence[Mapping[str, JsonValue]],
    converted: Sequence[tuple[tuple[ImportedItem, ...], bool]],
    ends: Sequence[int],
) -> tuple[int, ...]:
    """One turn end per assistant reply, not per record.

    A reply arrives as several records sharing the API message id -- thinking, then text, then each tool
    call -- and a continuation point in the middle of one would replay half an assistant turn.
    """
    return tuple(
        end
        for index, (end, (items, is_reply)) in enumerate(zip(ends, converted))
        if is_reply and items and _message_id(history[index]) != _next_reply_id(history, index)
    )


def _next_reply_id(history: Sequence[Mapping[str, JsonValue]], index: int) -> str | None:
    following: Final = history[index + 1] if index + 1 < len(history) else None
    return _message_id(following) if following is not None and following.get("type") == "assistant" else None


def _message_id(record: Mapping[str, JsonValue] | None) -> str | None:
    value: Final = _object((record or _EMPTY).get("message")).get("id")
    return str(value) if isinstance(value, str) else None


def _active_path(records: Sequence[Mapping[str, JsonValue]]) -> tuple[Mapping[str, JsonValue], ...]:
    """The history the harness itself would resume: the last record written, back to the root."""
    by_uuid: Final = MappingProxyType(
        {str(record["uuid"]): record for record in records if isinstance(record.get("uuid"), str)}
    )
    leaf: Final = next(
        (
            str(record["uuid"])
            for record in reversed(records)
            if isinstance(record.get("uuid"), str) and not record.get("isSidechain")
        ),
        None,
    )
    return tuple(reversed(tuple(_ancestors(by_uuid, leaf))))


def _ancestors(by_uuid: Mapping[str, Mapping[str, JsonValue]], leaf: str | None) -> Iterator[Mapping[str, JsonValue]]:
    seen: Final[set[str]] = set()  # mutable-ok: cycle guard while walking parent links
    node: str | None = leaf  # rebind-ok: walks from the leaf to the root
    while node is not None and node in by_uuid and node not in seen:
        seen.add(node)
        record: Mapping[str, JsonValue] = by_uuid[node]  # rebind-ok: one record per step
        yield record
        node = _parent_of(record)


def _parent_of(record: Mapping[str, JsonValue]) -> str | None:
    """A record's parent, across a compaction boundary.

    The boundary is written with no `parentUuid` -- from the model's side the conversation restarts there --
    and keeps the history it replaced on `logicalParentUuid`, which is what makes the dropped turns
    recoverable at all.
    """
    for key in ("parentUuid", "logicalParentUuid"):
        value: JsonValue = record.get(key)  # rebind-ok: one candidate link per key
        if isinstance(value, str):
            return value
    return None


def _record_items(record: Mapping[str, JsonValue], version: str) -> tuple[tuple[ImportedItem, ...], bool]:
    """A record's items, and whether it is part of an assistant reply."""
    kind: Final = str(record.get("type", ""))
    source: Final = MappingProxyType({"uuid": record.get("uuid"), "record_type": kind})
    if kind == "user":
        return _user_items(record, source), False
    if kind == "assistant":
        return _assistant_items(record, source, version), True
    if kind == "system":
        return _system_items(record, source), False
    if kind == "attachment":
        return _attachment_items(record, source), False
    return _NO_ITEMS, False


def _unrecognized(record: Mapping[str, JsonValue]) -> bool:
    return str(record.get("type", "")) not in ("user", "assistant", "system", "attachment")


def _note(text: str, kind: str, source: Meta) -> ImportedItem:
    return user_message((input_text(text),), MappingProxyType({"kind": kind, "source": source}))


def _user_items(record: Mapping[str, JsonValue], source: Meta) -> tuple[ImportedItem, ...]:
    """Tool results first, as their own items, then whatever the user actually said."""
    message: Final = _object(record.get("message"))
    content: Final = message.get("content")
    kind: Final = (
        "compaction_summary"
        if record.get("isCompactSummary")
        else "harness_context"
        if record.get("isMeta")
        else "user"
    )
    if isinstance(content, str):
        return (_note(content, kind, source),) if content else _NO_ITEMS
    blocks: Final = tuple(block for block in (content if isinstance(content, list) else ()) if isinstance(block, dict))
    outputs: Final = tuple(_tool_output(block, source) for block in blocks if block.get("type") == "tool_result")
    spoken: Final = tuple(item for block in blocks if (item := _user_block(block)) is not None)
    meta: Final = MappingProxyType({"kind": kind, "source": source})
    return outputs + ((user_message(spoken, meta),) if spoken else _NO_ITEMS)


def _user_block(block: Mapping[str, JsonValue]) -> Item | None:
    if block.get("type") == "text" and block.get("text"):
        return input_text(str(block["text"]))
    if block.get("type") == "image":
        return _image(_object(block.get("source")))
    return None


def _image(origin: Mapping[str, JsonValue]) -> Item | None:
    if origin.get("type") == "url" and isinstance(origin.get("url"), str):
        return input_image(str(origin["url"]))
    if isinstance(origin.get("data"), str):
        return input_image(f"data:{origin.get('media_type', 'image/png')};base64,{origin['data']}")
    return None


def _tool_output(block: Mapping[str, JsonValue], source: Meta) -> ImportedItem:
    error: Final = _ERROR_OUTPUT if block.get("is_error") else _TOOL_OUTPUT
    return function_call_output(
        str(block.get("tool_use_id", "")),
        _text(block.get("content")),
        MappingProxyType({**error, "source": source}),
    )


def _assistant_items(record: Mapping[str, JsonValue], source: Meta, version: str) -> tuple[ImportedItem, ...]:
    message: Final = _object(record.get("message"))
    provenance: Final = MappingProxyType(
        {
            "harness": SOURCE,
            "harness_version": version,
            "provider": "anthropic",
            "api": "anthropic_messages",
            "model": message.get("model"),
            "stop_reason": message.get("stop_reason"),
            "message_id": message.get("id"),
        }
    )
    meta: Final = MappingProxyType({"source": source, "provenance": provenance})
    blocks: Final = message.get("content")
    return tuple(
        item
        for block in (blocks if isinstance(blocks, list) else ())
        if isinstance(block, dict) and (item := _assistant_block(block, meta)) is not None
    )


def _assistant_block(block: Mapping[str, JsonValue], meta: Meta) -> ImportedItem | None:
    kind: Final = block.get("type")
    if kind == "text" and block.get("text"):
        return assistant_text(str(block["text"]), MappingProxyType({**meta, "kind": "text"}))
    if kind in ("thinking", "redacted_thinking"):
        return _thinking(block, meta)
    if kind == "tool_use":
        return function_call(
            str(block.get("id", "")),
            str(block.get("name", "")),
            block.get("input") or "{}",
            MappingProxyType({**meta, "kind": "tool_call"}),
        )
    return None


def _thinking(block: Mapping[str, JsonValue], meta: Meta) -> ImportedItem:
    """The signature travels as encrypted content, marked with the provider that can verify it.

    Anthropic will only accept thinking back with its own signature, so a view rendering this session for
    another provider drops it rather than sending something that would be rejected.
    """
    signature: Final = block.get("signature") or block.get("data")
    sealed: Final = str(signature) if isinstance(signature, str) and signature else None
    bound: Final = (
        MappingProxyType({**meta, "kind": "reasoning", "provider": "anthropic"})
        if sealed
        else MappingProxyType({**meta, "kind": "reasoning"})
    )
    return reasoning(str(block.get("thinking", "")), sealed, bound)


def _system_items(record: Mapping[str, JsonValue], source: Meta) -> tuple[ImportedItem, ...]:
    subtype: Final = str(record.get("subtype", ""))
    if subtype in _CONTROL_SUBTYPES:
        return _NO_ITEMS
    text: Final = _text(record.get("content"))
    if not text:
        return _NO_ITEMS
    if subtype == "compact_boundary":
        boundary: Final = MappingProxyType({**source, "compaction": record.get("compactMetadata")})
        return (_note(text, "compaction_boundary", boundary),)
    return (_note(text, "harness_context", MappingProxyType({**source, "subtype": subtype})),)


def _attachment_items(record: Mapping[str, JsonValue], source: Meta) -> tuple[ImportedItem, ...]:
    attachment: Final = _object(record.get("attachment"))
    kind: Final = str(attachment.get("type", ""))
    if kind in _EPHEMERAL_ATTACHMENTS:
        return _NO_ITEMS
    text: Final = _rendered(record.get("rendered"))
    if not text.strip():
        return _NO_ITEMS
    return (_note(text, "harness_context", MappingProxyType({**source, "attachment": kind})),)


def _rendered(blocks: JsonValue) -> str:
    """What the harness actually put in front of the model for an attachment.

    Each block carries the injected text under `content`; an attachment the harness chose not to render
    reaches the model as nothing, so it is stored as nothing.
    """
    if isinstance(blocks, str):
        return blocks
    return "\n".join(
        str(block.get("content") or block.get("text") or "")
        for block in (blocks if isinstance(blocks, list) else ())
        if isinstance(block, dict)
    )


def _title(records: Sequence[Mapping[str, JsonValue]]) -> str | None:
    for key in ("customTitle", "aiTitle"):
        titles: tuple[str, ...] = tuple(  # rebind-ok: one candidate per key
            str(record[key]) for record in records if isinstance(record.get(key), str)
        )
        if titles:
            return titles[-1]
    return _last_string(records, "slug") or None


def _last_string(records: Sequence[Mapping[str, JsonValue]], key: str) -> str:
    return next((str(record[key]) for record in reversed(records) if isinstance(record.get(key), str)), "")


def _last_model(records: Sequence[Mapping[str, JsonValue]]) -> str:
    return next(
        (
            str(model)
            for record in reversed(records)
            if record.get("type") == "assistant"
            and isinstance(model := _object(record.get("message")).get("model"), str)
        ),
        "",
    )


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
    return value if isinstance(value, dict) else _EMPTY


def _at(record: Mapping[str, JsonValue]) -> datetime:
    stamp: Final = record.get("timestamp")
    if not isinstance(stamp, str):
        return datetime.fromtimestamp(0, UTC)
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, UTC)
