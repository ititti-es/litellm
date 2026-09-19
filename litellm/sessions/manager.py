from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

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

    @property
    def budget_bytes(self) -> int:
        return self._budget

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
    ) -> Session:
        session: Final = self._next_session(owner, prior, turn_input, output, response_id, now, new_session_id)
        self._sessions[session.id] = session
        self._sessions.move_to_end(session.id)
        self._sizes[session.id] = session.size()
        self._by_response[response_id] = session.id
        self._commits += 1
        self._evict_over_budget(keep=session.id)
        return session

    def stats(self) -> ManagerStats:
        return ManagerStats(
            sessions=len(self._sessions),
            held_bytes=sum(self._sizes.values()),
            commits=self._commits,
            forks=self._forks,
            evictions=self._evictions,
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
