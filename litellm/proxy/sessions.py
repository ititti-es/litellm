from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import SessionCompactionSettings, SessionManagerSettings
from litellm.repositories.conversation_session_repository import (
    ConversationSessionRepository,
    ItemRow,
    StoredSession,
    TurnWrite,
    jsonb_text,
)
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.sessions import Item, ReplayBase, Session, SessionManager
from litellm.sessions.cache_clock import CacheClock
from litellm.sessions.compaction import DEFAULT, CallUpstream, CompactionPolicy, CompactionReply, compact, reply_from
from litellm.sessions.responses import MANAGED_BY_CALLER

if TYPE_CHECKING:
    from litellm.router import Router

_ITEM: Final = TypeAdapter(dict[str, JsonValue])
_RETENTION_SWEEP_SECONDS: Final = 3600
# Set by the compaction call itself; never taken from the deployment or the replayed request
_RESERVED: Final = frozenset({"input", "model", "custom_llm_provider", "stream", "litellm_metadata"})


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
    return SessionManager(budget_bytes=budget, clock=CacheClock(target=settings.compaction.target_hit_rate))


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
                    ItemRow(
                        seq=seq,
                        content=jsonb_text(item),
                        is_output=seq in session.outputs,
                        meta=jsonb_text(session.annotations[seq]) if seq in session.annotations else None,
                    )
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
        annotations: Final = MappingProxyType(
            {
                row.seq: _ITEM.validate_python(_decoded(row.meta))
                for row in stored.items
                if getattr(row, "meta", None) is not None
            }
        )
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
        annotations=annotations,
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


def router_upstream(
    router: Callable[[], Router | None], send: Callable[..., Awaitable[object]] | None = None
) -> CallUpstream:
    """Compaction calls reuse the deployment that served the turn, so they reach the same upstream cache"""
    responses: Final = send or litellm.aresponses

    async def call(replay: ReplayBase, items: Sequence[Item], owner: str) -> CompactionReply:
        current: Final = router()
        credentials: Final = (
            current.get_deployment_credentials_with_provider(replay.deployment)
            if current is not None and replay.deployment is not None
            else None
        )
        deployment: Final = MappingProxyType(credentials) if credentials is not None else MappingProxyType({})
        response: Final = await responses(
            input=items,
            **{name: value for name, value in deployment.items() if name not in _RESERVED},  # mutable-ok: call kwargs
            **{
                name: value for name, value in replay.params.items() if name not in _RESERVED
            },  # mutable-ok: call kwargs
            custom_llm_provider=deployment.get("custom_llm_provider", replay.provider),
            model=replay.model,
            stream=False,
            litellm_metadata=_compaction_metadata(owner),
        )
        return reply_from(response)

    return call


def _compaction_metadata(owner: str) -> dict[str, object]:
    """Marked as the manager's own call, so it is not tracked as a turn, and attributed to the session's key"""
    return {MANAGED_BY_CALLER: True, "user_api_key_hash": owner, "tags": ["session_compaction"]}  # mutable-ok: metadata


async def load_policy(repository: ConversationSessionRepository | None, name: str | None) -> CompactionPolicy:
    """The configured stored policy, or the built-in one when none is named or it cannot be read"""
    if repository is None or name is None:
        return DEFAULT
    try:
        stored: Final = await repository.find_policy(name)
    except Exception as exc:
        verbose_proxy_logger.warning("session manager: could not load compaction policy %s: %s", name, exc)
        return DEFAULT
    return CompactionPolicy(stored.name, stored.instructions) if stored is not None else DEFAULT


async def compact_before_expiry(
    manager: SessionManager,
    settings: SessionCompactionSettings,
    call: CallUpstream,
    repository: ConversationSessionRepository | None,
) -> None:
    running: Final[set[asyncio.Task[Session | None]]] = set()  # mutable-ok: strong refs to in-flight compactions
    while True:
        await asyncio.sleep(settings.tick_seconds)
        for task in await _start_due(manager, settings, call, repository, time.time()):
            running.add(task)
            task.add_done_callback(running.discard)


async def _start_due(
    manager: SessionManager,
    settings: SessionCompactionSettings,
    call: CallUpstream,
    repository: ConversationSessionRepository | None,
    now: float,
) -> tuple[asyncio.Task[Session | None], ...]:
    jobs: Final = manager.due_compactions(now, settings.min_prefix_tokens, settings.lead_seconds)
    if not jobs:
        return ()
    policy: Final = await load_policy(repository, settings.policy)
    return tuple(asyncio.create_task(compact(manager, job, policy, call)) for job in jobs)


def start_compaction(
    manager: SessionManager | None,
    raw_settings: object,
    repository: ConversationSessionRepository | None,
    router: Callable[[], Router | None],
) -> asyncio.Task[None] | None:
    if manager is None or raw_settings is None:
        return None
    settings: Final = SessionManagerSettings.model_validate(raw_settings).compaction
    if not settings.enabled:
        return None
    return asyncio.create_task(compact_before_expiry(manager, settings, router_upstream(router), repository))
