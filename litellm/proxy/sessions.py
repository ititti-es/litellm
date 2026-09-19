from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SessionManagerSettings
from litellm.repositories.conversation_session_repository import (
    ConversationSessionRepository,
    ItemRow,
    StoredSession,
    TurnWrite,
    jsonb_text,
)
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.sessions import Item, Session, SessionManager

_ITEM: Final = TypeAdapter(dict[str, JsonValue])
_RETENTION_SWEEP_SECONDS: Final = 3600


def configured_session_manager(raw: object, current: SessionManager | None) -> SessionManager | None:
    """The manager general_settings.session_manager asks for, keeping the live one (and its sessions) across reloads"""
    if raw is None:
        return None
    settings: Final = SessionManagerSettings.model_validate(raw)
    if not settings.enabled:
        return None
    budget: Final = settings.budget_mb << 20
    if current is not None and current.budget_bytes == budget:
        return current
    return SessionManager(budget_bytes=budget)


class PrismaTranscriptStore:
    def __init__(self, repository: ConversationSessionRepository, retention: timedelta | None) -> None:
        self._repository: Final = repository
        self._retention: Final = retention

    async def save(self, session: Session, first_new: int, response_id: str) -> None:
        now: Final = datetime.now(UTC)
        await self._repository.save_turn(
            TurnWrite(
                session_id=session.id,
                owner=session.owner,
                version=session.version,
                last_upstream_at=datetime.fromtimestamp(session.last_upstream_at, UTC),
                expires_at=now + self._retention if self._retention is not None else None,
                items=tuple(
                    ItemRow(seq=seq, content=jsonb_text(item), is_output=seq in session.outputs)
                    for seq, item in enumerate(session.items)
                    if seq >= first_new
                ),
                response_id=response_id,
                item_count=session.responses[response_id],
            )
        )

    async def load(self, response_id: str) -> Session | None:
        stored: Final = await self._repository.load_by_response(response_id)
        return _session_from(stored) if stored is not None else None


def _session_from(stored: StoredSession) -> Session | None:
    """None unless the stored rows form the complete transcript, so a partial write never resumes a conversation"""
    if tuple(row.seq for row in stored.items) != tuple(range(len(stored.items))):
        return None
    try:
        items: Final[tuple[Item, ...]] = tuple(_ITEM.validate_python(_decoded(row.content)) for row in stored.items)
    except ValidationError:
        return None
    return Session(
        id=stored.session_id,
        owner=stored.owner,
        items=items,
        outputs=frozenset(row.seq for row in stored.items if row.is_output),
        responses=MappingProxyType({row.response_id: row.item_count for row in stored.responses}),
        version=stored.version,
        last_upstream_at=stored.last_upstream_at.timestamp(),
    )


def _decoded(content: object) -> object:
    return json.loads(content) if isinstance(content, str) else content


def attach_transcript_store(
    manager: SessionManager | None, raw_settings: object, prisma_client: DatabaseClient | None
) -> ConversationSessionRepository | None:
    if manager is None or prisma_client is None or raw_settings is None:
        return None
    settings: Final = SessionManagerSettings.model_validate(raw_settings)
    if not settings.store_transcripts:
        return None
    repository: Final = ConversationSessionRepository(prisma_client)
    retention: Final = timedelta(days=settings.retention_days) if settings.retention_days is not None else None
    manager.attach_store(PrismaTranscriptStore(repository, retention))
    return repository


async def sweep_expired_transcripts(repository: ConversationSessionRepository) -> None:
    while True:
        await asyncio.sleep(_RETENTION_SWEEP_SECONDS)
        await _sweep_once(repository)


async def _sweep_once(repository: ConversationSessionRepository) -> None:
    try:
        deleted: Final = await repository.delete_expired(datetime.now(UTC))
        verbose_proxy_logger.debug("session manager: deleted %d expired transcripts", deleted)
    except Exception as exc:
        verbose_proxy_logger.exception("session manager: retention sweep failed: %s", exc)
