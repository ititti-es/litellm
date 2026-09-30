from __future__ import annotations

import asyncio
import hashlib
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

from litellm._logging import verbose_logger
from litellm.sessions.cache_clock import CacheClock, CacheEstimate, retention_hint
from litellm.sessions.session import Item, Meta, Session, comparable


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
class Continuation:
    """Where the next turn of a client that keeps its own history belongs.

    `prior` is the stored conversation this turn continues, or None when it begins one; `turn_input` is what
    is actually new, which for a client that resends everything is the tail of what it sent.
    """

    session_id: str
    prior: Resolved | None
    turn_input: tuple[Item, ...]


def client_session_key(owner: str, client_session_id: str) -> str:
    """The session id a client-named conversation is stored at.

    Derived rather than looked up, so the next turn finds the conversation with no index to keep and no id to
    hand back, and a gateway restart finds it too. Hashed with the owner because the client chooses the value:
    two API keys naming the same session must land on different rows, and one key must not be able to guess
    its way to another's. Distinct from the upstream cache key, which is derived separately from the same two.
    """
    digest: Final = hashlib.sha256(f"session\0{owner}\0{client_session_id}".encode()).hexdigest()
    return f"cs_{digest[:32]}"


@dataclass(frozen=True, slots=True)
class ManagerStats:
    sessions: int
    held_bytes: int
    commits: int
    forks: int
    evictions: int
    rehydrations: int
    unpersisted: int


@dataclass(frozen=True, slots=True)
class TurnUsage:
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ReplayBase:
    """What an upstream request needs besides its input to reuse a turn's cached prefix; held in memory only"""

    provider: str
    model: str
    params: Mapping[str, object]
    deployment: str | None = None
    compaction_policy: str | None = None
    """The policy the client asked for (request metadata `compaction_policy`), if any"""
    model_group: str | None = None
    """The proxy's public model name, which per-model compaction settings are keyed by"""


@dataclass(frozen=True, slots=True)
class TurnRecord:
    response_id: str
    started_at: float
    usage: TurnUsage | None
    replay: ReplayBase | None
    begun_at: float
    compacted: bool = False
    compactable: bool = True


@dataclass(frozen=True, slots=True)
class CompactionJob:
    session: Session
    response_id: str
    replay: ReplayBase
    history: tuple[Item, ...]
    tail_start: int
    warm_until: float
    turn_started_at: float
    prefix_tokens: int
    """What the last turn read (its input), the yardstick for whether its cache survived"""
    next_prefix_tokens: int
    """What the next turn would send: the last turn's input and output"""


class TranscriptStore(Protocol):
    async def save(self, session: Session, first_new: int, response_id: str) -> None: ...

    async def load(self, response_id: str) -> Session | None: ...

    async def load_session(self, session_id: str) -> Session | None:
        """The conversation stored under this id, for a client that names its session rather than its turn"""
        ...


@dataclass(frozen=True, slots=True)
class _PendingWrite:
    session: Session
    first_new: int
    response_id: str
    settled: asyncio.Future[bool] | None = None
    """Resolved True once stored, False if the store refused it; what `persisted` waits on"""


# Write outcomes kept for `persisted`; a turn nobody waits on (a websocket or sync stream) ages out
_SETTLED_KEPT: Final = 1024


class SessionManager:
    """Owns every conversation's transcript, addressed by the response ids the gateway has returned

    Held sessions are bounded by `budget_bytes`; the least recently used ones are evicted first, and an
    evicted session resolves as Unknown, which callers treat exactly as today's stateless path
    """

    def __init__(
        self,
        budget_bytes: int = 256 << 20,
        clock: CacheClock | None = None,
        record_store_false: bool = False,
        refuse_unknown_previous_response: bool = True,
    ) -> None:
        self._budget: Final = budget_bytes
        self._clock: Final = clock or CacheClock()
        self.record_store_false = record_store_false  # writable-ok: a config reload may change it
        """Record `store: false` turns without rewriting them; see SessionManagerSettings.record_store_false"""
        self.refuse_unknown_previous_response = refuse_unknown_previous_response  # writable-ok: reloadable
        """See SessionManagerSettings.refuse_unknown_previous_response"""
        self._turns: Final[dict[str, TurnRecord]] = {}  # mutable-ok: latest upstream turn per held session
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
        self._settled: Final[OrderedDict[str, asyncio.Future[bool]]] = OrderedDict()  # mutable-ok: by response id

    def attach_store(self, store: TranscriptStore) -> None:
        self._store = store

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def clock(self) -> CacheClock:
        return self._clock

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

    async def continue_client_session(self, owner: str, client_session_id: str, items: Sequence[Item]) -> Continuation:
        """Where a turn belongs that names its session but not the turn it follows.

        A harness that keeps its own history (LangGraph, and anything else built on a local checkpointer)
        sends the whole conversation every turn and no `previous_response_id`. Taken at face value each turn
        is a new conversation, and the store fills with one session per turn, each a superset of the last.

        So the session is found from the client's own id, and the turn is accepted as its continuation only
        when what arrived actually extends what is stored: the gateway replays its own copy of the history,
        and a caller can only ever extend a transcript it can already reproduce. Anything else starts a
        conversation of its own rather than overwriting one.
        """
        key: Final = client_session_key(owner, client_session_id)
        stored: Final = await self._client_session(key)
        if stored is None:
            return Continuation(key, None, tuple(items))
        continued: Final = _continues(stored, items)
        if continued is None or stored.owner != owner:
            return Continuation(uuid.uuid4().hex, None, tuple(items))
        response_id, tail = continued
        return Continuation(stored.id, Resolved(stored, response_id), tuple(items)[tail:])

    async def _client_session(self, session_id: str) -> Session | None:
        held: Final = self._sessions.get(session_id)
        if held is not None:
            self._sessions.move_to_end(session_id)
            return held
        if self._store is None:
            return None
        loaded: Final = await self._store.load_session(session_id)
        if loaded is None:
            return None
        self._hold(loaded)
        self._rehydrations += 1
        return loaded

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
        started_at: float | None = None,
        usage: TurnUsage | None = None,
        replay: ReplayBase | None = None,
        client_session_id: str | None = None,
        compactable: bool = True,
    ) -> Session:
        session: Final = self._next_session(
            owner, prior, turn_input, output, response_id, now, new_session_id, client_session_id
        )
        held: Final = self._sessions.get(session.id)
        first_new: Final = len(held.items) if held is not None and held.version == session.version else 0
        started: Final = started_at if started_at is not None else now
        self._observe(prior, started, usage, replay)
        self._hold(session)
        self._turns[session.id] = TurnRecord(
            response_id, started, usage, replay, begun_at=started, compactable=compactable
        )
        self._commits += 1
        if durable and self._store is not None:
            self._persist(_PendingWrite(session, first_new, response_id))
        return session

    def last_turn(self, session_id: str) -> TurnRecord | None:
        return self._turns.get(session_id)

    def begin(self, session_id: str, now: float) -> None:
        """A turn on this session is on its way upstream; compaction must not race it"""
        record: Final = self._turns.get(session_id)
        if record is not None:
            self._turns[session_id] = replace(record, begun_at=now)

    def _observe(
        self, prior: Resolved | None, started_at: float, usage: TurnUsage | None, replay: ReplayBase | None
    ) -> None:
        last: Final = self._turns.get(prior.session.id) if prior is not None else None
        if (
            prior is None
            or last is None
            or last.compacted
            or last.response_id != prior.response_id
            or last.usage is None
            or last.replay is None
            or usage is None
            or replay is None
            or (last.replay.provider, last.replay.model) != (replay.provider, replay.model)
        ):
            return
        self._clock.observe(
            replay.provider,
            replay.model,
            started_at - last.started_at,
            last.usage.input_tokens,
            usage.cached_tokens,
            observed_at=started_at,
        )

    def observe_compaction(self, job: CompactionJob, started_at: float, usage: TurnUsage) -> None:
        """A compaction reads the prefix at the very gap being predicted, under current load: first-hand evidence"""
        self._clock.observe(
            job.replay.provider,
            job.replay.model,
            started_at - job.turn_started_at,
            job.prefix_tokens,
            usage.cached_tokens,
            observed_at=started_at,
        )

    def cache_estimate(self, session_id: str) -> CacheEstimate | None:
        record: Final = self._turns.get(session_id)
        session: Final = self._sessions.get(session_id)
        if record is None or record.replay is None or session is None:
            return None
        return self._clock.estimate(record.replay.provider, record.replay.model, retention_hint(record.replay.params))

    def warm_until(self, session_id: str) -> float | None:
        """When this session's cached prefix is predicted to go cold"""
        record: Final = self._turns.get(session_id)
        estimate: Final = self.cache_estimate(session_id)
        if record is None or estimate is None or estimate.ttl_seconds <= 0:
            return None
        return record.started_at + estimate.ttl_seconds

    def due_compactions(self, now: float, min_prefix_tokens: int, lead_seconds: float) -> tuple[CompactionJob, ...]:
        """Sessions whose warm window is about to close, large enough to be worth compacting while it is open"""
        return tuple(
            job
            for session_id in tuple(self._turns)
            if (job := self._due(session_id, now, min_prefix_tokens, lead_seconds))
        )

    def _due(self, session_id: str, now: float, min_prefix_tokens: int, lead_seconds: float) -> CompactionJob | None:
        record: Final = self._turns.get(session_id)
        session: Final = self._sessions.get(session_id)
        deadline: Final = self.warm_until(session_id)
        if (
            record is None
            or session is None
            or deadline is None
            or record.compacted
            or not record.compactable
            or record.replay is None
            or record.usage is None
            or record.begun_at > record.started_at
            or record.usage.input_tokens + record.usage.output_tokens < min_prefix_tokens
            or not deadline - lead_seconds <= now < deadline
            or session.responses.get(record.response_id) != len(session.items)
        ):
            return None
        tail: Final = compaction_tail(session)
        if tail is None:
            return None
        self._turns[session_id] = replace(record, compacted=True)
        return CompactionJob(
            session=session,
            response_id=record.response_id,
            replay=record.replay,
            history=session.replayable_at(record.response_id),
            tail_start=tail,
            warm_until=deadline,
            turn_started_at=record.started_at,
            prefix_tokens=record.usage.input_tokens,
            next_prefix_tokens=record.usage.input_tokens + record.usage.output_tokens,
        )

    def apply_compaction(
        self, job: CompactionJob, summary: Sequence[Item], summary_at: int, summary_meta: Meta, now: float
    ) -> Session | None:
        """Rebase the session onto its summary items plus the kept tail; None when a turn landed meanwhile"""
        live: Final = self._sessions.get(job.session.id)
        if live is None or live.version != job.session.version or len(live.items) != len(job.session.items):
            return None
        pending: Final = self._turns.get(live.id)
        if pending is not None and pending.begun_at > pending.started_at:
            return None
        head: Final = len(summary)
        items: Final = (*summary, *live.items[job.tail_start :])
        compacted: Final = Session(
            id=live.id,
            owner=live.owner,
            items=items,
            outputs=frozenset(
                (
                    *(index for index, item in enumerate(summary) if item.get("role") == "assistant"),
                    *(index - job.tail_start + head for index in live.outputs if index >= job.tail_start),
                )
            ),
            responses=MappingProxyType({job.response_id: len(items)}),
            version=live.version + 1,
            last_upstream_at=now,
            annotations=_rebased_annotations(live.annotations, job.tail_start, head, summary_at, summary_meta),
            client_session_id=live.client_session_id,
        )
        self._hold(compacted)
        record: Final = self._turns.get(live.id)
        if record is not None:
            self._turns[live.id] = replace(record, usage=None, compacted=True)
        if self._store is not None:
            self._persist(_PendingWrite(compacted, 0, job.response_id))
        return compacted

    async def flush(self) -> None:
        if self._writes is not None:
            await self._writes.join()

    async def persisted(self, response_id: str, timeout: float) -> bool:
        """Whether the turn that returned *response_id* is in the durable store, waiting up to *timeout* for it

        True when there is no store to wait for. False when the write failed, did not finish in time, or was
        never queued (an in-memory-only commit, or an outcome old enough to have aged out).
        """
        if self._store is None:
            return True
        settled: Final = self._settled.get(response_id)
        if settled is None:
            return False
        try:
            return await asyncio.wait_for(asyncio.shield(settled), timeout)
        except TimeoutError:
            return False

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
        settled: Final[asyncio.Future[bool]] = loop.create_future()
        self._settled[write.response_id] = settled
        self._settled.move_to_end(write.response_id)
        while len(self._settled) > _SETTLED_KEPT:
            self._settled.popitem(last=False)
        self._writes.put_nowait(replace(write, settled=settled))

    async def _drain(self, writes: asyncio.Queue[_PendingWrite]) -> None:
        while True:
            write: _PendingWrite = await writes.get()  # rebind-ok: one pending write per loop iteration
            stored = False  # rebind-ok: one outcome per write
            try:
                if self._store is not None:
                    await self._store.save(write.session, write.first_new, write.response_id)
                stored = True
            except Exception as exc:
                self._unpersisted += 1
                verbose_logger.exception("SessionManager: failed to persist session %s: %s", write.session.id, exc)
            finally:
                if write.settled is not None and not write.settled.done():
                    write.settled.set_result(stored)
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
        client_session_id: str | None,
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
                client_session_id=client_session_id,
            )
        base: Final = self._sessions.get(prior.session.id, prior.session)
        end: Final = prior.session.responses[prior.response_id]
        if end == len(base.items) and not base.superseded:
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
                base.annotations,
                client_session_id or base.client_session_id,
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
            MappingProxyType({index: meta for index, meta in base.annotations.items() if index < end}),
            client_session_id or base.client_session_id,
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
        self._turns.pop(session_id, None)
        for response_id in session.responses:
            if self._by_response.get(response_id) == session_id:
                del self._by_response[response_id]


def _continues(session: Session, items: Sequence[Item]) -> tuple[str, int] | None:
    """The response this input continues from and where its new items start, or None if it is another story"""
    latest: Final = _latest_response(session)
    if latest is None:
        return None
    response_id, end = latest
    stored: Final = tuple(form for _, form in comparable(session.items[:end]))
    arrived: Final = comparable(items)
    if len(arrived) <= len(stored) or tuple(form for _, form in arrived[: len(stored)]) != stored:
        return None
    return response_id, arrived[len(stored)][0]


def _latest_response(session: Session) -> tuple[str, int] | None:
    """The turn that ends the stored history; a session mid-fork has none that covers all of it"""
    complete: Final = tuple(rid for rid, end in session.responses.items() if end == len(session.items))
    return (complete[0], len(session.items)) if complete else None


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
    annotations: Mapping[int, Meta] | None = None,
    client_session_id: str | None = None,
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
        annotations=annotations if annotations is not None else MappingProxyType({}),
        client_session_id=client_session_id,
    )


_NO_META: Final[Meta] = MappingProxyType({})


def _rebased_annotations(
    annotations: Mapping[int, Meta], tail_start: int, head: int, summary_at: int, summary: Meta
) -> Mapping[int, Meta]:
    """The summary's meta at its position, and the kept tail's meta moved to follow the summary items"""
    shift: Final = head - tail_start
    kept: Final = MappingProxyType({i + shift: m for i, m in annotations.items() if i >= tail_start})
    return MappingProxyType({summary_at: summary, **kept})  # mutable-ok: frozen on construction


def compaction_tail(session: Session, keep_turns: int = 1) -> int | None:
    """Where the kept tail starts: the *keep_turns*-th last user message, so the latest exchanges stay verbatim"""
    starts: Final = tuple(
        index
        for index, item in enumerate(session.items)
        if index not in session.outputs
        and item.get("type", "message") == "message"
        and item.get("role") == "user"
        and session.annotations.get(index, _NO_META).get("kind", "user") == "user"
    )
    start: Final = starts[-keep_turns] if len(starts) >= keep_turns else 0
    return start if start > 0 else None
