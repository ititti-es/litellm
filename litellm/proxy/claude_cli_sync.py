"""Keeping the transcript store level with Claude Code's own record of a `claude_cli` conversation.

The provider reports a finished turn; this reads Claude Code's transcript for that session and imports it,
which is the same path `session_import` takes for an archive. Re-importing an already-stored session appends
only what is new, so running this after every turn costs one read of a file the harness just wrote.

It runs after the answer has gone back to the caller and never reports failure to them: a transcript that
could not be stored is a gap in the record, not a failed request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.llms.claude_cli.sync import SyncRequest, set_transcript_sync
from litellm.proxy.session_import import run_import
from litellm.repositories.conversation_session_repository import ConversationSessionRepository
from litellm.sessions.importers.claude_code import ImportedSession, read_claude_code_session

_LOCKS: Final[dict[str, _Waiters]] = {}  # mutable-ok: one lock per session, so its turns import in order
_RUNNING: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: holds tasks so they are not collected mid-run


def attach_claude_cli_sync(repository: ConversationSessionRepository | None) -> bool:
    """Store Claude Code's transcripts wherever the session manager stores its own. Returns whether it will."""
    if repository is None:
        set_transcript_sync(None)
        return False
    set_transcript_sync(_scheduler(repository))
    return True


async def drain_claude_cli_sync() -> None:
    """Wait for the transcripts still being stored, so a shutdown does not drop the last turn's record."""
    while _RUNNING:
        await asyncio.gather(*tuple(_RUNNING), return_exceptions=True)


def _scheduler(repository: ConversationSessionRepository) -> Callable[[SyncRequest], None]:
    def schedule(request: SyncRequest) -> None:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            # A synchronous completion has no loop to hand this to. The proxy always has one; a library
            # caller doing `litellm.completion` gets the answer and no stored transcript.
            verbose_proxy_logger.debug("No event loop to sync the Claude Code transcript for %s", request.session_id)
            return
        task: Final = loop.create_task(sync_session(request, repository))
        _RUNNING.add(task)
        task.add_done_callback(_RUNNING.discard)

    return schedule


async def sync_session(request: SyncRequest, repository: ConversationSessionRepository) -> None:
    async with _serialised(request.session_id):
        try:
            await _sync(request, repository)
        except Exception as exc:  # noqa: BLE001  # the turn is already served; this is only the record of it
            verbose_proxy_logger.warning(
                "Could not store the Claude Code transcript for %s: %s", request.session_id, exc
            )


async def _sync(request: SyncRequest, repository: ConversationSessionRepository) -> None:
    projects: Final = Path(request.projects_dir) if request.projects_dir else None
    session: Final = await asyncio.to_thread(read_claude_code_session, request.session_id, projects)
    if session is None:
        verbose_proxy_logger.debug("Claude Code kept no transcript for session %s", request.session_id)
        return
    for line in await run_import((_linked(session, request),), request.owner, repository):
        verbose_proxy_logger.debug("claude_cli transcript: %s", line)


def _linked(session: ImportedSession, request: SyncRequest) -> ImportedSession:
    """Record which conversation of the caller's this transcript is, so the store shares their identity."""
    return replace(
        session,
        metadata=MappingProxyType(
            {**session.metadata, "litellm_session_id": request.client_session_id, "provider": "claude_cli"}
        ),
    )


@dataclass(slots=True)
class _Waiters:
    lock: asyncio.Lock
    holders: int


@asynccontextmanager
async def _serialised(session_id: str) -> AsyncGenerator[None]:
    """Import one session's turns one at a time; concurrently they would each see a half-written history.

    The lock is dropped once nobody holds it, so a gateway serving many short conversations does not
    accumulate one per session for the life of the process.
    """
    entry: Final = _LOCKS.setdefault(session_id, _Waiters(asyncio.Lock(), 0))
    entry.holders += 1
    try:
        async with entry.lock:
            yield
    finally:
        entry.holders -= 1
        if entry.holders == 0:
            _LOCKS.pop(session_id, None)
