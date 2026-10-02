"""opencode sessions (SQLite `session`, `message`, `part`) as harness-neutral transcripts"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from itertools import accumulate, groupby
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
    loads,
    reasoning,
    sqlite_snapshot,
    user_message,
)
from litellm.sessions.session import Item

SOURCE: Final = "opencode"
_CONTROL_PARTS: Final = frozenset({"step-start", "step-finish", "snapshot", "patch", "agent", "retry"})
_INTERRUPTED: Final = "[Tool execution was interrupted]"
_COMPACTION_REQUEST: Final = "Summarize the conversation so far."
_TOOL_OUTPUT: Final[Meta] = MappingProxyType({"kind": "tool_output"})
_PRUNED_OUTPUT: Final[Meta] = MappingProxyType({"kind": "tool_output", "pruned_by_harness": True})
_ERROR_OUTPUT: Final[Meta] = MappingProxyType({"kind": "tool_output", "is_error": True})
_INTERRUPTED_OUTPUT: Final[Meta] = MappingProxyType({"kind": "interrupted_tool"})


def read_opencode(db_path: Path, scratch_dir: Path | None = None) -> Iterator[ImportedSession]:
    """Either store format: the v1 `session`/`message`/`part` tables, or v2 `session_v2` with `session_message`"""
    with sqlite_snapshot(db_path, scratch_dir) as db:
        tables: Final = frozenset(
            str(name) for (name,) in db.execute("select name from sqlite_master where type='table'")
        )
        v1_rows: Final = "session" in tables and db.execute("select count(*) from session").fetchone()[0] > 0
        if not v1_rows and "session_v2" in tables:
            yield from _v2_sessions(db, db_path.name)
            return
        sessions: Final = db.execute(
            "select id, parent_id, title, version, revert, time_created, time_updated, directory "
            "from session order by time_created, id"
        ).fetchall()
        for row in sessions:
            yield _session(db, row)


def _session(db: sqlite3.Connection, row: tuple[object, ...]) -> ImportedSession:
    session_id, parent_id, title, version, revert, created, updated, directory = row
    messages: Final = db.execute(
        "select id, data from message where session_id = ? order by time_created, id", (session_id,)
    ).fetchall()
    part_rows: Final = db.execute(
        "select message_id, id, data from part where session_id = ? order by message_id, id", (session_id,)
    ).fetchall()
    parts: Final = MappingProxyType(
        {message_id: tuple(group) for message_id, group in groupby(part_rows, key=lambda part: part[0])}
    )
    kept: Final = _before_revert(messages, loads(revert))
    per_message: Final = tuple(
        _message_items(str(message_id), _object(loads(data)), parts.get(message_id, ()), str(version or ""))
        for message_id, data in kept
    )
    ends: Final = tuple(accumulate(len(items) for items, _, _ in per_message))
    return ImportedSession(
        source=SOURCE,
        source_id=str(session_id),
        title=str(title) if title else None,
        parent_source_id=str(parent_id) if parent_id else None,
        created_at=_ms(created),
        updated_at=_ms(updated),
        items=tuple(item for items, _, _ in per_message for item in items),
        turn_ends=tuple(end for end, (items, is_turn, _) in zip(ends, per_message) if is_turn and items),
        metadata=MappingProxyType(
            {"harness": SOURCE, "harness_version": str(version or ""), "directory": str(directory or "")}
        ),
        skipped=sum(skipped for _, _, skipped in per_message),
    )


def _before_revert(messages: Sequence[tuple[object, ...]], revert: JsonValue) -> Sequence[tuple[object, ...]]:
    """opencode deletes everything from the reverted message onwards at its next prompt"""
    cut: Final = revert.get("messageID") if isinstance(revert, dict) else None
    ids: Final = [str(row[0]) for row in messages]  # mutable-ok: local lookup list
    return messages[: ids.index(cut)] if isinstance(cut, str) and cut in ids else messages


def _message_items(
    message_id: str, info: Mapping[str, JsonValue], parts: Sequence[tuple[object, ...]], version: str
) -> tuple[tuple[ImportedItem, ...], bool, int]:
    """The message's items, whether it is an assistant turn, and how many unrecognized parts it skipped"""
    decoded: Final = tuple((str(part_id), _object(loads(data))) for _, part_id, data in parts)
    if info.get("role") == "user":
        return _user_items(message_id, decoded), False, _unknown(decoded, _USER_PARTS)
    if _skipped_error(info, decoded):
        return (), False, 0
    return _assistant_items(message_id, info, decoded, version), True, _unknown(decoded, _ASSISTANT_PARTS)


_USER_PARTS: Final = frozenset({"text", "file", "compaction", "subtask"}) | _CONTROL_PARTS
_ASSISTANT_PARTS: Final = frozenset({"text", "reasoning", "tool"}) | _CONTROL_PARTS


def _unknown(parts: Sequence[tuple[str, Mapping[str, JsonValue]]], known: frozenset[str]) -> int:
    return sum(1 for _, part in parts if part.get("type") not in known)


def _user_items(message_id: str, parts: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> tuple[ImportedItem, ...]:
    content: Final = tuple(block for part_id, part in parts if (block := _user_block(part)) is not None)
    source: Final = MappingProxyType(
        {"message_id": message_id, "parts": tuple(MappingProxyType({**part, "id": part_id}) for part_id, part in parts)}
    )
    requests: Final = tuple(
        user_message(
            (input_text(_COMPACTION_REQUEST),),
            MappingProxyType({"kind": "compaction_request", "source": MappingProxyType({"part_id": part_id})}),
        )
        for part_id, part in parts
        if part.get("type") == "compaction"
    )
    subtasks: Final = tuple(
        user_message(
            (input_text(str(part.get("prompt", ""))),),
            MappingProxyType({"kind": "subtask", "source": MappingProxyType({**part, "id": part_id})}),
        )
        for part_id, part in parts
        if part.get("type") == "subtask"
    )
    typed: Final = (user_message(content, MappingProxyType({"kind": "user", "source": source})),) if content else ()
    return (*typed, *subtasks, *requests)


def _user_block(part: Mapping[str, JsonValue]) -> Item | None:
    kind: Final = part.get("type")
    if kind == "text" and not part.get("ignored") and part.get("text"):
        return input_text(str(part["text"]))
    if kind != "file":
        return None
    mime: Final = str(part.get("mime", ""))
    if mime.startswith("image/"):
        return input_image(str(part.get("url", "")))
    if mime in ("text/plain", "application/x-directory"):
        return None
    return input_text(f"[file: {part.get('filename') or part.get('url')} ({mime})]")


def _skipped_error(info: Mapping[str, JsonValue], parts: Sequence[tuple[str, Mapping[str, JsonValue]]]) -> bool:
    """opencode's own rule: an errored turn is dropped unless it was aborted after producing real content"""
    error: Final = info.get("error")
    if not error:
        return False
    aborted: Final = isinstance(error, dict) and error.get("name") == "MessageAbortedError"
    return not (aborted and any(part.get("type") not in ("step-start", "reasoning") for _, part in parts))


def _assistant_items(
    message_id: str, info: Mapping[str, JsonValue], parts: Sequence[tuple[str, Mapping[str, JsonValue]]], version: str
) -> tuple[ImportedItem, ...]:
    provenance: Final = MappingProxyType(
        {
            "harness": SOURCE,
            "harness_version": version,
            "provider": info.get("providerID"),
            "model": info.get("modelID"),
            "agent": info.get("agent") or info.get("mode"),
            "variant": info.get("variant"),
        }
    )
    summary: Final = bool(info.get("summary"))
    return tuple(
        item for part_id, part in parts for item in _assistant_part(part, part_id, message_id, provenance, summary)
    )


def _assistant_part(
    part: Mapping[str, JsonValue], part_id: str, message_id: str, provenance: Meta, summary: bool
) -> tuple[ImportedItem, ...]:
    source: Final = MappingProxyType({"message_id": message_id, "part": MappingProxyType({**part, "id": part_id})})
    kind: Final = part.get("type")
    if kind == "text" and part.get("text"):
        label: Final = "compaction_summary" if summary else "text"
        return (
            assistant_text(
                str(part["text"]), MappingProxyType({"kind": label, "source": source, "provenance": provenance})
            ),
        )
    if kind == "reasoning" and part.get("text") is not None:
        encrypted: Final = _encrypted_content(part.get("metadata"))
        meta: Final = MappingProxyType({"kind": "reasoning", "source": source, "provenance": provenance})
        bound: Final = MappingProxyType({**meta, "provider": provenance.get("provider")}) if encrypted else meta
        return (reasoning(str(part["text"]), encrypted, bound),)
    if kind == "tool":
        return _tool_items(part, source, provenance)
    return ()


def _tool_items(part: Mapping[str, JsonValue], source: Meta, provenance: Meta) -> tuple[ImportedItem, ...]:
    state: Final = _object(part.get("state"))
    call_id: Final = str(part.get("callID", ""))
    call: Final = function_call(
        call_id,
        str(part.get("tool", "")),
        state.get("input") or "{}",
        MappingProxyType({"kind": "tool_call", "source": source, "provenance": provenance}),
    )
    status: Final = state.get("status")
    time: Final = _object(state.get("time"))
    if status == "completed":
        meta: Final = _PRUNED_OUTPUT if time.get("compacted") else _TOOL_OUTPUT
        return (call, function_call_output(call_id, str(state.get("output", "")), meta))
    if status == "error":
        metadata: Final = _object(state.get("metadata"))
        output: Final = (
            str(metadata.get("output", "")) if metadata.get("interrupted") else f"Error: {state.get('error', '')}"
        )
        return (call, function_call_output(call_id, output, _ERROR_OUTPUT))
    return (call, function_call_output(call_id, _INTERRUPTED, _INTERRUPTED_OUTPUT))


def _encrypted_content(metadata: JsonValue) -> str | None:
    if not isinstance(metadata, dict):
        return None
    for value in metadata.values():
        if isinstance(value, dict):
            found: JsonValue = value.get("reasoningEncryptedContent") or value.get(
                "encrypted_content"
            )  # rebind-ok: one provider entry per iteration
            if isinstance(found, str) and found:
                return found
    return None


def _object(value: JsonValue) -> Mapping[str, JsonValue]:
    return value if isinstance(value, dict) else MappingProxyType({})


def _ms(value: object) -> datetime:
    return datetime.fromtimestamp((value if isinstance(value, int | float) else 0) / 1000, UTC)


def _v2_sessions(db: sqlite3.Connection, store: str) -> Iterator[ImportedSession]:
    sessions: Final = db.execute(
        "select id, parent_id, title, version, revert, time_created, time_updated, directory "
        "from session_v2 order by time_created, id"
    ).fetchall()
    for row in sessions:
        yield _v2_session(db, row, store)


def _v2_session(db: sqlite3.Connection, row: tuple[object, ...], store: str) -> ImportedSession:
    session_id, parent_id, title, version, revert, created, updated, directory = row
    messages: Final = _before_revert(
        db.execute(
            "select id, type, data from session_message where session_id = ? order by seq", (session_id,)
        ).fetchall(),
        loads(revert),
    )
    per_message: Final = tuple(
        _v2_message_items(str(message_id), MappingProxyType({**_object(loads(data)), "type": kind}), str(version or ""))
        for message_id, kind, data in messages
    )
    ends: Final = tuple(accumulate(len(items) for items, _, _ in per_message))
    return ImportedSession(
        source=SOURCE,
        source_id=str(session_id),
        title=str(title) if title else None,
        parent_source_id=str(parent_id) if parent_id else None,
        created_at=_ms(created),
        updated_at=_ms(updated),
        items=tuple(item for items, _, _ in per_message for item in items),
        turn_ends=tuple(end for end, (items, is_turn, _) in zip(ends, per_message) if is_turn and items),
        metadata=MappingProxyType(
            {
                "harness": SOURCE,
                "harness_version": str(version or ""),
                "directory": str(directory or ""),
                "format": "v2",
                "store": store,
            }
        ),
        skipped=sum(skipped for _, _, skipped in per_message),
    )


_V2_CONTROL: Final = frozenset({"agent-switched", "model-switched", "idle", "location-switched"})
_V2_NOTE_KINDS: Final = MappingProxyType(
    {"synthetic": "harness_context", "system": "system_note", "shell": "shell_note"}
)


def _v2_message_items(
    message_id: str, message: Mapping[str, JsonValue], version: str
) -> tuple[tuple[ImportedItem, ...], bool, int]:
    """The message's items, whether it is an assistant turn, and whether it was an unrecognized type"""
    kind: Final = message.get("type")
    source: Final = MappingProxyType({"message_id": message_id, "message": message})
    if kind == "user":
        content: Final = _v2_user_content(message)
        return (
            (user_message(content, MappingProxyType({"kind": "user", "source": source})),) if content else (),
            False,
            0,
        )
    if kind == "assistant":
        blocks: Final = message.get("content")
        if message.get("error") and not blocks:
            return (), False, 0
        return _v2_assistant_items(message, source, version), True, 0
    if kind in ("synthetic", "system", "shell", "compaction"):
        return (_v2_note(message, str(kind), source),), False, 0
    return (), False, 0 if kind in _V2_CONTROL else 1


def _v2_user_content(message: Mapping[str, JsonValue]) -> tuple[Item, ...]:
    text: Final = message.get("text")
    files: Final = message.get("files")
    attachments: Final = tuple(
        _v2_attachment(file) for file in (files if isinstance(files, list) else ()) if isinstance(file, dict)
    )
    return (*((input_text(str(text)),) if text else ()), *attachments)


def _v2_attachment(file: Mapping[str, JsonValue]) -> Item:
    """A v2 attachment names its file by `uri`, or carries it inline as base64 `data` (source type "inline")"""
    mime: Final = str(file.get("mime", ""))
    uri: Final = file.get("uri")
    data: Final = file.get("data")
    url: Final = (
        uri
        if isinstance(uri, str) and uri
        else f"data:{mime};base64,{data}"
        if isinstance(data, str) and data
        else ""
    )
    if mime.startswith("image/") and url:
        return input_image(url)
    return input_text(f"[file: {file.get('name') or uri or 'inline'} ({mime or 'unknown type'})]")


def _v2_note(message: Mapping[str, JsonValue], kind: str, source: Meta) -> ImportedItem:
    """Harness-authored turns stored as what they are; how each harness frames them is a view"""
    text: Final = (
        f"Ran `{message.get('command', '')}`\n{message.get('output', '')}"
        if kind == "shell"
        else str(message.get("summary", ""))
        if kind == "compaction"
        else str(message.get("text", ""))
    )
    label: Final = _V2_NOTE_KINDS.get(kind, "compaction_summary")
    return user_message((input_text(text),), MappingProxyType({"kind": label, "source": source}))


def _v2_assistant_items(message: Mapping[str, JsonValue], source: Meta, version: str) -> tuple[ImportedItem, ...]:
    model: Final = _object(message.get("model"))
    provider_state: Final = _object(message.get("providerState"))
    provenance: Final = MappingProxyType(
        {
            "harness": SOURCE,
            "harness_version": version,
            "provider": model.get("providerID"),
            "model": model.get("id"),
            "variant": model.get("variant"),
            "agent": message.get("agent"),
            "provider_response_id": provider_state.get("responseId"),
        }
    )
    blocks: Final = message.get("content")
    return tuple(
        item
        for block in (blocks if isinstance(blocks, list) else ())
        if isinstance(block, dict)
        for item in _v2_block(block, source, provenance)
    )


def _v2_block(block: Mapping[str, JsonValue], source: Meta, provenance: Meta) -> tuple[ImportedItem, ...]:
    kind: Final = block.get("type")
    meta: Final = MappingProxyType({"source": source, "block_id": block.get("id"), "provenance": provenance})
    if kind == "text" and block.get("text"):
        return (assistant_text(str(block["text"]), MappingProxyType({**meta, "kind": "text"})),)
    if kind == "reasoning":
        encrypted: Final = _encrypted_content(block.get("providerMetadata"))
        tagged: Final = MappingProxyType({**meta, "kind": "reasoning", "provider": provenance.get("provider")})
        return (
            reasoning(
                str(block.get("text", "")),
                encrypted,
                tagged if encrypted else MappingProxyType({**meta, "kind": "reasoning"}),
            ),
        )
    if kind != "tool":
        return ()
    state: Final = _object(block.get("state"))
    call_id: Final = str(block.get("id", ""))
    call: Final = function_call(
        call_id, str(block.get("name", "")), state.get("input") or "{}", MappingProxyType({**meta, "kind": "tool_call"})
    )
    status: Final = state.get("status")
    if status == "completed":
        pruned: Final = _object(block.get("time")).get("pruned")
        return (
            call,
            function_call_output(call_id, _tool_text(state.get("content")), _PRUNED_OUTPUT if pruned else _TOOL_OUTPUT),
        )
    if status == "error":
        error: Final = _object(state.get("error"))
        return (call, function_call_output(call_id, f"Error: {error.get('message', '')}", _ERROR_OUTPUT))
    return (call, function_call_output(call_id, _INTERRUPTED, _INTERRUPTED_OUTPUT))


def _tool_text(content: JsonValue) -> str:
    return "\n".join(
        str(block.get("text", ""))
        if block.get("type") == "text"
        else f"[file: {block.get('name') or block.get('uri')}]"
        for block in (content if isinstance(content, list) else ())
        if isinstance(block, dict)
    )
