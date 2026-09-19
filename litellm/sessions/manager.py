from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from litellm._logging import verbose_logger
from litellm.sessions.session import Item, Session


@dataclass(frozen=True, slots=True)
class Resolved:
    session: Session
    response_id: str

    @property
    def history(self) -> tuple[Item, ...]:
        return self.session.replayable_at(self.response_id)


@dataclass(frozen=True, slots=True)
class Unknown:
    response_id: str


@dataclass(frozen=True, slots=True)
class Forbidden:
    response_id: str


Resolution: TypeAlias = Resolved | Unknown | Forbidden


@dataclass(frozen=True, slots=True)
class ManagerStats:
    sessions: int
    held_bytes: int
    commits: int
    forks: int
    evictions: int
    rehydrations: int
    unpersisted: int


class TranscriptStore(Protocol):
    async def save(self, session: Session, first_new: int, response_id: str) -> None: ...

    async def load(self, response_id: str) -> Session | None: ...


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    session: Session
    first_new: int
    response_id: str


class SessionManager:
    """Owns every conversation's transcript, addressed by the response ids the gateway has returned

    Held sessions are bounded by `budget_bytes`; the least recently used ones are evicted first, and an
    evicted session resolves as Unknown, which callers treat exactly as today's stateless path
    """

    def __init__(self, budget_bytes: int = 256 << 20) -> None:
        self._budget: Final = budget_bytes
        self._sessions: Final[OrderedDict[str, Session]] = OrderedDict()  # mutable-ok: LRU order of held sessions
        self._sizes: Final[dict[str, int]] = {}  # mutable-ok: byte size per held session, kept beside the LRU
        self._by_response: Final[dict[str, str]] = {}  # mutable-ok: response id to session id index
        self._commits = 0
        self._forks = 0
        self._evictions = 0
        self._rehydrations = 0
        self._unpersisted = 0
        self._store: TranscriptStore | None = None
        self._writes: asyncio.Queue[_PendingWrite] | None = None
        self._writer: asyncio.Task[None] | None = None

    def attach_store(self, store: TranscriptStore) -> None:
        self._store = store

    @property
    def budget_bytes(self) -> int:
        return self._budget

    async def resolve_async(self, response_id: str, owner: str) -> Resolution:
        """Like resolve, rehydrating from the durable store when the session is no longer held"""
        held: Final = self.resolve(response_id, owner)
        if not isinstance(held, Unknown) or self._store is None:
            return held
        loaded: Final = await self._store.load(response_id)
        if loaded is None:
            return held
        self._hold(loaded)
        self._rehydrations += 1
        return self.resolve(response_id, owner)

    def resolve(self, response_id: str, owner: str) -> Resolution:
        session_id: Final = self._by_response.get(response_id)
        session: Final = self._sessions.get(session_id) if session_id is not None else None
        if session is None or response_id not in session.responses:
            return Unknown(response_id)
        if session.owner != owner:
            return Forbidden(response_id)
        self._sessions.move_to_end(session.id)
        return Resolved(session, response_id)

    def commit(
        self,
        *,
        owner: str,
        prior: Resolved | None,
        turn_input: Sequence[Item],
        output: Sequence[Item],
        response_id: str,
        now: float,
        new_session_id: str | None = None,
        durable: bool = True,
    ) -> Session:
        session: Final = self._next_session(owner, prior, turn_input, output, response_id, now, new_session_id)
        held: Final = self._sessions.get(session.id)
        first_new: Final = len(held.items) if held is not None and held.version == session.version else 0
        self._hold(session)
        self._commits += 1
        if durable and self._store is not None:
            self._persist(_PendingWrite(session, first_new, response_id))
        return session

    async def flush(self) -> None:
        if self._writes is not None:
            await self._writes.join()

    def _hold(self, session: Session) -> None:
        self._sessions[session.id] = session
        self._sessions.move_to_end(session.id)
        self._sizes[session.id] = session.size()
        for response_id in session.responses:
            self._by_response[response_id] = session.id
        self._evict_over_budget(keep=session.id)

    def _persist(self, write: _PendingWrite) -> None:
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            self._unpersisted += 1
            return
        if self._writes is None:
            self._writes = asyncio.Queue()
            self._writer = loop.create_task(self._drain(self._writes))
        self._writes.put_nowait(write)

    async def _drain(self, writes: asyncio.Queue[_PendingWrite]) -> None:
        while True:
            write: _PendingWrite = await writes.get()  # rebind-ok: one pending write per loop iteration
            try:
                if self._store is not None:
                    await self._store.save(write.session, write.first_new, write.response_id)
            except Exception as exc:
                self._unpersisted += 1
                verbose_logger.exception("SessionManager: failed to persist session %s: %s", write.session.id, exc)
            finally:
                writes.task_done()

    def stats(self) -> ManagerStats:
        return ManagerStats(
            sessions=len(self._sessions),
            held_bytes=sum(self._sizes.values()),
            commits=self._commits,
            forks=self._forks,
            evictions=self._evictions,
            rehydrations=self._rehydrations,
            unpersisted=self._unpersisted,
        )

    def _next_session(
        self,
        owner: str,
        prior: Resolved | None,
        turn_input: Sequence[Item],
        output: Sequence[Item],
        response_id: str,
        now: float,
        new_session_id: str | None,
    ) -> Session:
        if prior is None:
            return _built(
                new_session_id or uuid.uuid4().hex,
                owner,
                (),
                frozenset(),
                MappingProxyType({}),
                0,
                turn_input,
                output,
                response_id,
                now,
            )
        base: Final = self._sessions.get(prior.session.id, prior.session)
        end: Final = prior.session.responses[prior.response_id]
        if end == len(base.items):
            return _built(
                base.id,
                owner,
                base.items,
                base.outputs,
                base.responses,
                base.version,
                turn_input,
                output,
                response_id,
                now,
            )
        self._forks += 1
        return _built(
            uuid.uuid4().hex,
            owner,
            base.items[:end],
            frozenset(i for i in base.outputs if i < end),
            MappingProxyType({rid: n for rid, n in base.responses.items() if n <= end}),
            0,
            turn_input,
            output,
            response_id,
            now,
        )

    def _evict_over_budget(self, keep: str) -> None:
        while sum(self._sizes.values()) > self._budget and len(self._sessions) > 1:
            oldest: str = next(iter(self._sessions))  # rebind-ok: walks the LRU from its cold end
            if oldest == keep:
                self._sessions.move_to_end(keep)
                continue
            self._drop(oldest)
            self._evictions += 1

    def _drop(self, session_id: str) -> None:
        session: Final = self._sessions.pop(session_id)
        self._sizes.pop(session_id, None)
        for response_id in session.responses:
            if self._by_response.get(response_id) == session_id:
                del self._by_response[response_id]


def _built(
    session_id: str,
    owner: str,
    items: tuple[Item, ...],
    outputs: frozenset[int],
    responses: Mapping[str, int],
    version: int,
    turn_input: Sequence[Item],
    output: Sequence[Item],
    response_id: str,
    now: float,
) -> Session:
    new_items: Final = (*items, *turn_input, *output)
    first_output: Final = len(items) + len(turn_input)
    return Session(
        id=session_id,
        owner=owner,
        items=new_items,
        outputs=outputs | frozenset(range(first_output, len(new_items))),
        responses=MappingProxyType({**responses, response_id: len(new_items)}),
        version=version,
        last_upstream_at=now,
    )
