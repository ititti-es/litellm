"""Import harness sessions into the session manager's transcript store

    python -m litellm.proxy.session_import opencode [--source ~/.local/share/opencode/opencode.db] [--owner-key sk-...]
    python -m litellm.proxy.session_import openclaw [--source ~/.openclaw] [--owner-key sk-...]
    python -m litellm.proxy.session_import claude-code [--source ~/.claude/projects] [--owner-key sk-...]
    python -m litellm.proxy.session_import pin|unpin SESSION_ID

Sources are read from a private copy, never written. Re-running is safe: a session that grew since the last
import is appended to, and one whose history the harness rewrote is stored as a new version beside the old,
and a session a key is already using keeps that owner rather than being handed to the importer's.
Without --owner-key, imports are archive-only: stored, but continuable by no key.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import AsyncGenerator, Iterable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm.proxy._types import hash_token
from litellm.repositories.conversation_session_repository import (
    ConversationSessionRepository,
    ImportWrite,
    ItemRow,
    StoredTranscript,
    jsonb_text,
)
from litellm.sessions.importers.claude_code import DEFAULT_PROJECTS_DIR, read_claude_code
from litellm.sessions.importers.common import ImportedSession
from litellm.sessions.importers.openclaw import read_openclaw
from litellm.sessions.importers.opencode import read_opencode
from litellm.sessions.session import canonical

_DEFAULT_SOURCES: Final = MappingProxyType(
    {
        "opencode": Path.home() / ".local" / "share" / "opencode" / "opencode.db",
        "openclaw": Path(os.environ.get("OPENCLAW_STATE_DIR", Path.home() / ".openclaw")),
        "claude-code": Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects"
        if os.environ.get("CLAUDE_CONFIG_DIR")
        else DEFAULT_PROJECTS_DIR,
    }
)

_READERS: Final = MappingProxyType(
    {"opencode": read_opencode, "openclaw": read_openclaw, "claude-code": read_claude_code}
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def imported_session_id(source: str, source_id: str) -> str:
    return f"imp_{source}_{_digest(source_id)}"


def imported_response_id(source: str, source_id: str, turn: int) -> str:
    return f"resp_imp_{_digest(f'{source}:{source_id}:{turn}')}"


@dataclass(frozen=True, slots=True)
class Planned:
    write: ImportWrite
    action: str


ARCHIVE_OWNER_PREFIX: Final = "archive:"


def holder(requested: str, stored: StoredTranscript | None) -> str:
    """Who owns the session after this import.

    An import never takes a session away from the key already using it: the `claude_cli` provider stores a
    conversation under the key it was billed to, and archiving the same session later must not leave that
    caller forbidden from their own history. An archive placeholder owns nothing and is adopted freely.
    """
    held: Final = stored.owner if stored is not None else None
    if not held or held.startswith(ARCHIVE_OWNER_PREFIX):
        return requested
    return held


def plan_import(session: ImportedSession, owner: str, stored: StoredTranscript | None) -> Planned:
    """Append when the stored transcript is a prefix of the source, otherwise store the source as a new version"""
    contents: Final = tuple(canonical(json.loads(jsonb_text(item.item))) for item in session.items)
    held: Final = tuple(canonical(_decoded(content)) for content in stored.contents) if stored is not None else ()
    extends: Final = stored is not None and contents[: len(held)] == held
    unchanged: Final = extends and len(held) == len(contents)
    version: Final = 0 if stored is None else stored.version if extends else stored.version + 1
    first_new: Final = len(held) if extends else 0
    session_id: Final = imported_session_id(session.source, session.source_id)
    write: Final = ImportWrite(
        session_id=session_id,
        owner=holder(owner, stored),
        version=version,
        source=session.source,
        source_id=session.source_id,
        title=session.title,
        parent_session_id=(
            imported_session_id(session.source, session.parent_source_id) if session.parent_source_id else None
        ),
        metadata=jsonb_text(MappingProxyType({**session.metadata, "skipped": session.skipped})),
        created_at=session.created_at,
        last_upstream_at=session.updated_at,
        items=tuple(
            ItemRow(
                seq=seq, content=jsonb_text(imported.item), is_output=imported.is_output, meta=jsonb_text(imported.meta)
            )
            for seq, imported in enumerate(session.items)
            if seq >= first_new
        ),
        responses=tuple(
            (imported_response_id(session.source, session.source_id, turn), end)
            for turn, end in enumerate(session.turn_ends)
        ),
    )
    change: Final = (
        "new" if stored is None else "unchanged" if unchanged else "appended" if extends else f"new version {version}"
    )
    kept: Final = "" if write.owner == owner else " (kept owner)"
    return Planned(write, change + kept)


def _decoded(content: object) -> object:
    return json.loads(content) if isinstance(content, str) else content


async def run_import(
    sessions: Iterable[ImportedSession],
    owner: str,
    repository: ConversationSessionRepository | None,
    write: bool = True,
) -> Iterator[str]:
    lines: Final = tuple([await _import_one(session, owner, repository, write) for session in sessions])
    return iter(lines)


async def _import_one(
    session: ImportedSession, owner: str, repository: ConversationSessionRepository | None, write: bool
) -> str:
    stored: Final = (
        await repository.current_transcript(imported_session_id(session.source, session.source_id))
        if repository is not None
        else None
    )
    planned: Final = plan_import(session, owner, stored)
    if write and repository is not None and not planned.action.startswith("unchanged"):
        await repository.import_session(planned.write)
    title: Final = (session.title or "")[:60]
    return (
        f"{planned.write.session_id}  {planned.action:<13} items={len(session.items):<5} "
        f"turns={len(session.turn_ends):<4} skipped={session.skipped:<3} {title}"
    )


@dataclass(frozen=True, slots=True)
class _Client:
    db: object


@asynccontextmanager
async def repository_from_url(url: str | None) -> AsyncGenerator[ConversationSessionRepository | None]:
    if url is None:
        yield None
        return
    from prisma import Prisma

    client: Final = Prisma(datasource={"url": url})  # mutable-ok: the Prisma constructor takes a dict
    await client.connect()
    try:
        yield ConversationSessionRepository(_Client(client))
    finally:
        await client.disconnect()


async def _main(args: argparse.Namespace) -> int:
    dry_run: Final = bool(getattr(args, "dry_run", False))
    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if url is None and not dry_run:
        sys.stderr.write("DATABASE_URL (or --database-url) is required unless --dry-run\n")
        return 2
    async with repository_from_url(url) as repository:
        if args.command in ("pin", "unpin"):
            if repository is None:
                return 2
            found: Final = await repository.set_pinned(args.session_id, args.command == "pin")
            sys.stdout.write(f"{args.session_id}: {args.command + 'ned' if found else 'not found'}\n")
            return 0 if found else 1
        source: Final = Path(args.source).expanduser() if args.source else _DEFAULT_SOURCES[args.command]
        owner: Final = hash_token(args.owner_key) if args.owner_key else f"{ARCHIVE_OWNER_PREFIX}{args.command}"
        reader: Final = _READERS[args.command]
        scratch: Final = Path(args.scratch_dir).expanduser() if args.scratch_dir else None
        for line in await run_import(reader(source, scratch), owner, repository, write=not dry_run):
            sys.stdout.write(line + "\n")
        return 0


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.session_import", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    for name in _READERS:
        sub = commands.add_parser(name, help=f"import {name} sessions")  # rebind-ok: one subcommand per iteration
        sub.add_argument("--source", help=f"defaults to {_DEFAULT_SOURCES[name]}")
        sub.add_argument("--owner-key", help="the virtual key that may continue these sessions")
        sub.add_argument("--dry-run", action="store_true", help="read and convert, write nothing")
        sub.add_argument(
            "--scratch-dir", help="read a private copy made here instead of the live file (read-only either way)"
        )
    for name in ("pin", "unpin"):
        sub = commands.add_parser(name, help=f"{name} a stored session")  # rebind-ok: one subcommand per iteration
        sub.add_argument("session_id")
    return parser


def main() -> int:
    return asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
