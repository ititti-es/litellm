"""Harness-neutral imported transcripts

Items are plain Responses API input items, the neutral core any harness or provider can be rendered from.
Everything harness-specific (the original record, how it was framed, which harness and model produced it,
which toolset a call came from) lives beside the item in its meta, so no view is lost and none is baked in.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import JsonValue

from litellm.sessions.session import Item

Meta: TypeAlias = Mapping[str, object]

_TOOL_CLASSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "bash": "shell",
        "shell": "shell",
        "exec": "shell",
        "local_shell": "shell",
        "execute_command": "shell",
        "read": "read_file",
        "read_file": "read_file",
        "view": "read_file",
        "write": "write_file",
        "write_file": "write_file",
        "create_file": "write_file",
        "edit": "edit_file",
        "multiedit": "edit_file",
        "edit_file": "edit_file",
        "str_replace_based_edit_tool": "edit_file",
        "apply_patch": "patch",
        "patch": "patch",
        "list": "list_dir",
        "ls": "list_dir",
        "list_dir": "list_dir",
        "grep": "search",
        "glob": "search",
        "find": "search",
        "search": "search",
        "codesearch": "search",
        "webfetch": "web_fetch",
        "web_fetch": "web_fetch",
        "fetch": "web_fetch",
        "websearch": "web_search",
        "web_search": "web_search",
        "xai_web_search": "web_search",
        "task": "subagent",
        "agent": "subagent",
        "sessions_spawn": "subagent",
        "todowrite": "todo",
        "todoread": "todo",
        "todo_write": "todo",
        "update_plan": "todo",
    }
)


def tool_class(name: str) -> str:
    return _TOOL_CLASSES.get(name.lower(), "other")


@dataclass(frozen=True, slots=True)
class ImportedItem:
    item: Item
    is_output: bool
    meta: Meta


@dataclass(frozen=True, slots=True)
class ImportedSession:
    source: str
    source_id: str
    title: str | None
    parent_source_id: str | None
    created_at: datetime
    updated_at: datetime
    items: tuple[ImportedItem, ...]
    turn_ends: tuple[int, ...]
    metadata: Meta
    skipped: int
    first_turn: int = 0
    """The turn number of `turn_ends[0]`: above zero once the turns before a compaction were cut away"""


def from_last_compaction(session: ImportedSession) -> ImportedSession:
    """The session as its harness last sent it: from its latest compaction (request, then summary) onward

    A harness that compacted never sends the history before its summary again, and that history can be far larger
    than the model's context window, so a session meant to be continued starts here, the way the session manager
    rebases its own compactions. Kept turns keep their numbers, so each keeps its response id; turns that ended
    before the cut belong to the version that holds the whole history.
    """
    summaries: Final = [
        index for index, item in enumerate(session.items) if item.meta.get("kind") == "compaction_summary"
    ]
    if not summaries:
        return session
    last: Final = summaries[-1]
    lead_in: Final = last > 0 and session.items[last - 1].meta.get("kind") == "compaction_request"
    cut: Final = last - 1 if lead_in else last
    kept: Final = [(turn, end - cut) for turn, end in enumerate(session.turn_ends) if end > cut]
    if not kept:
        return session
    return replace(
        session,
        items=session.items[cut:],
        turn_ends=tuple(end for _, end in kept),
        first_turn=session.first_turn + kept[0][0],
    )


def user_message(content: tuple[Item, ...], meta: Meta) -> ImportedItem:
    item: Final = {"type": "message", "role": "user", "content": list(content)}  # mutable-ok: Responses item JSON
    return ImportedItem(item, False, meta)


def input_text(text: str) -> Item:
    return {"type": "input_text", "text": text}  # mutable-ok: Responses content JSON


def input_image(url: str) -> Item:
    return {"type": "input_image", "image_url": url}  # mutable-ok: Responses content JSON


def assistant_text(text: str, meta: Meta) -> ImportedItem:
    content: Final = [{"type": "output_text", "text": text}]  # mutable-ok: Responses content JSON
    item: Final = {"type": "message", "role": "assistant", "content": content}  # mutable-ok: Responses item JSON
    return ImportedItem(item, True, meta)


def reasoning(summary: str, encrypted_content: str | None, meta: Meta) -> ImportedItem:
    """Kept for the record either way; replay drops reasoning that carries no encrypted content"""
    parts: Final = [{"type": "summary_text", "text": summary}] if summary else []  # mutable-ok: Responses JSON
    item: Final = {"type": "reasoning", "summary": parts}  # mutable-ok: Responses item JSON
    sealed: Final = {**item, "encrypted_content": encrypted_content}  # mutable-ok: Responses item JSON
    return ImportedItem(sealed if encrypted_content else item, True, meta)


def function_call(call_id: str, name: str, arguments: JsonValue, meta: Meta) -> ImportedItem:
    encoded: Final = arguments if isinstance(arguments, str) else json.dumps(arguments)
    item: Final = {"type": "function_call", "call_id": call_id, "name": name, "arguments": encoded}  # mutable-ok: JSON
    tool: Final = MappingProxyType({"name": name, "class": tool_class(name)})
    return ImportedItem(item, True, MappingProxyType({**meta, "tool": tool}))


def function_call_output(call_id: str, output: str, meta: Meta) -> ImportedItem:
    item: Final = {"type": "function_call_output", "call_id": call_id, "output": output}  # mutable-ok: Responses JSON
    return ImportedItem(item, False, meta)


@contextmanager
def sqlite_snapshot(path: Path, scratch_dir: Path | None = None) -> Generator[sqlite3.Connection]:
    """A read-only connection that never writes to the harness's store

    By default the live file is opened read-only (recent writes in its WAL included) and temporary sort data is
    kept in memory, so an import needs no free disk. With a scratch_dir, a private copy is read instead, which
    also isolates the import from a harness writing to the file at the same time
    """
    with (
        tempfile.TemporaryDirectory(prefix="litellm-import-", dir=scratch_dir) if scratch_dir else _no_copy() as scratch
    ):
        target: Final = Path(scratch) / path.name if scratch else path
        if scratch:
            for suffix in ("", "-wal", "-shm"):
                source: Path = path.with_name(path.name + suffix)  # rebind-ok: one sidecar file per iteration
                if source.exists():
                    shutil.copy2(source, target.with_name(target.name + suffix))
        connection: Final = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        connection.execute("pragma temp_store = memory")
        try:
            yield connection
        finally:
            connection.close()


@contextmanager
def _no_copy() -> Generator[str]:
    yield ""


def loads(raw: object) -> JsonValue:
    return json.loads(raw) if isinstance(raw, str | bytes) else None
