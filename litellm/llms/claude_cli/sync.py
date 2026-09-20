"""Telling whoever owns the transcript store that a Claude Code session has moved on.

Claude Code holds the conversation this provider serves: on a resumed session only the newest message is
sent, and the reply comes back as one string, so the proxy sees neither the history nor anything the CLI did
to produce the answer. The record of the turn exists, but in Claude Code's own transcript.

This is the seam between the two. The provider knows a session advanced and says so; the proxy, which is the
part that has a database, registers what to do about it. Nothing is registered by default, so a CLI login or
a library call runs exactly as before.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_logger


@dataclass(frozen=True, slots=True)
class SyncRequest:
    """A Claude Code session that has just produced a turn."""

    session_id: str
    """Claude Code's own session id, which names its transcript file"""
    owner: str
    """The key the turn was billed to, so the imported session is continuable by whoever produced it"""
    client_session_id: str | None
    """The caller's `litellm_session_id`, which links the stored transcript to the client's own identity"""
    projects_dir: str | None
    """Where this deployment's Claude Code keeps its transcripts, if not the default"""


TranscriptSync = Callable[[SyncRequest], None]

_SYNC: Final[list[TranscriptSync]] = []  # mutable-ok: single-slot registry


def set_transcript_sync(sync: TranscriptSync | None) -> None:
    _SYNC.clear()
    if sync is not None:
        _SYNC.append(sync)


def sync_transcript(request: SyncRequest) -> None:
    """Report a finished turn. Never raises: a turn that was served is not a failure because of bookkeeping."""
    if not _SYNC:
        return
    try:
        _SYNC[0](request)
    except Exception as exc:  # noqa: BLE001  # the answer is already the caller's; this is only the record of it
        verbose_logger.warning("Could not sync the Claude Code transcript for %s: %s", request.session_id, exc)
