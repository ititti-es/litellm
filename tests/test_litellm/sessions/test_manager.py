import asyncio
from dataclasses import replace
from types import MappingProxyType

from litellm.sessions import Forbidden, Resolved, SessionManager, Unknown
from litellm.sessions.manager import client_session_key

USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
ASSISTANT = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "hello", "annotations": []}],
}
CALL = {"type": "function_call", "call_id": "c1", "name": "ls", "arguments": "{}"}
CALL_OUTPUT = {"type": "function_call_output", "call_id": "c1", "output": "a b"}
REASONING_PLAIN = {"type": "reasoning", "id": "rs_1", "summary": []}
REASONING_SEALED = {"type": "reasoning", "id": "rs_2", "summary": [], "encrypted_content": "opaque"}


def first_turn(manager: SessionManager, owner: str = "key-a", response_id: str = "r1"):
    return manager.commit(
        owner=owner, prior=None, turn_input=(USER,), output=(REASONING_PLAIN, CALL), response_id=response_id, now=1.0
    )


def test_continuation_replays_history_in_the_shape_a_stateless_upstream_accepts():
    manager = SessionManager()
    first_turn(manager)
    prior = manager.resolve("r1", "key-a")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="key-a",
        prior=prior,
        turn_input=(CALL_OUTPUT,),
        output=(REASONING_SEALED, ASSISTANT),
        response_id="r2",
        now=2.0,
    )
    resolved = manager.resolve("r2", "key-a")
    assert isinstance(resolved, Resolved)
    assert resolved.history == (
        USER,
        CALL,
        CALL_OUTPUT,
        REASONING_SEALED,
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
    )
    assert resolved.session.items[-1] == ASSISTANT
    assert manager.stats().sessions == 1


def test_replayed_tool_calls_pair_with_their_outputs_without_the_reasoning_they_followed():
    """An `fc_` id without its reasoning item, or a dropped custom tool call, is a 400 upstream"""
    manager = SessionManager()
    custom = {"type": "custom_tool_call", "id": "ctc_1", "call_id": "c2", "name": "apply_patch", "input": "x"}
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(USER,),
        output=(REASONING_PLAIN, {**CALL, "id": "fc_1"}, custom),
        response_id="r1",
        now=1.0,
    )
    resolved = manager.resolve("r1", "k")
    assert isinstance(resolved, Resolved)
    assert resolved.history == (USER, CALL, {key: value for key, value in custom.items() if key != "id"})


def test_input_items_are_never_projected_even_when_they_look_like_output():
    manager = SessionManager()
    manager.commit(owner="k", prior=None, turn_input=(REASONING_PLAIN, ASSISTANT), output=(), response_id="r1", now=1.0)
    resolved = manager.resolve("r1", "k")
    assert isinstance(resolved, Resolved)
    assert resolved.history == (REASONING_PLAIN, ASSISTANT)


def test_continuing_from_an_older_response_forks_without_touching_the_original():
    manager = SessionManager()
    first_turn(manager)
    r1 = manager.resolve("r1", "key-a")
    assert isinstance(r1, Resolved)
    manager.commit(owner="key-a", prior=r1, turn_input=(CALL_OUTPUT,), output=(ASSISTANT,), response_id="r2", now=2.0)
    branch = manager.commit(
        owner="key-a",
        prior=r1,
        turn_input=({"role": "user", "content": "other"},),
        output=(),
        response_id="r3",
        now=3.0,
    )
    original = manager.resolve("r2", "key-a")
    assert isinstance(original, Resolved)
    assert branch.id != original.session.id
    assert branch.items[:2] == original.session.items[:2]
    assert CALL_OUTPUT not in branch.items
    assert manager.stats().forks == 1


def test_another_owner_cannot_continue_a_session():
    manager = SessionManager()
    first_turn(manager)
    assert manager.resolve("r1", "key-b") == Forbidden("r1")
    assert manager.resolve("nope", "key-a") == Unknown("nope")


def test_least_recently_used_session_is_evicted_first_and_then_resolves_unknown():
    one_session = first_turn(SessionManager()).size()
    manager = SessionManager(budget_bytes=one_session * 2 + one_session // 2)
    first_turn(manager, response_id="a")
    first_turn(manager, response_id="b")
    assert isinstance(manager.resolve("a", "key-a"), Resolved)
    first_turn(manager, response_id="c")
    assert manager.resolve("b", "key-a") == Unknown("b")
    assert isinstance(manager.resolve("a", "key-a"), Resolved)
    assert isinstance(manager.resolve("c", "key-a"), Resolved)
    assert manager.stats().evictions == 1


class MemoryStore:
    def __init__(self, fail: bool = False):
        self.saved: list = []
        self.sessions: dict = {}
        self.fail = fail

    async def save(self, session, first_new, response_id):
        if self.fail:
            raise ConnectionError("db down")
        self.saved.append((session.id, first_new, response_id))
        self.sessions[response_id] = session

    async def load(self, response_id):
        return self.sessions.get(response_id)

    async def load_session(self, session_id):
        saved = [s for s in self.sessions.values() if s.id == session_id]
        return saved[-1] if saved else None

    async def client_sessions(self, owner, client_session_id, limit):
        ids = dict.fromkeys(
            s.id for s in self.sessions.values() if s.owner == owner and s.client_session_id == client_session_id
        )
        return tuple(reversed(ids))[:limit]


async def test_durable_commits_write_only_the_new_items_in_order():
    store = MemoryStore()
    manager = SessionManager()
    manager.attach_store(store)
    first = first_turn(manager)
    prior = manager.resolve("r1", "key-a")
    manager.commit(
        owner="key-a", prior=prior, turn_input=(CALL_OUTPUT,), output=(ASSISTANT,), response_id="r2", now=2.0
    )
    manager.commit(owner="key-a", prior=None, turn_input=(USER,), output=(), response_id="r3", now=3.0, durable=False)
    await manager.flush()
    assert store.saved == [(first.id, 0, "r1"), (first.id, 3, "r2")]


async def test_an_evicted_session_is_rehydrated_from_the_store_for_its_owner_only():
    store = MemoryStore()
    one_session = first_turn(SessionManager()).size()
    manager = SessionManager(budget_bytes=one_session + one_session // 2)
    manager.attach_store(store)
    first_turn(manager, response_id="a")
    first_turn(manager, response_id="b")
    await manager.flush()
    assert manager.resolve("a", "key-a") == Unknown("a")

    assert await manager.resolve_async("a", "key-b") == Forbidden("a")
    resolved = await manager.resolve_async("a", "key-a")
    assert isinstance(resolved, Resolved) and resolved.history[0] == USER
    assert manager.stats().rehydrations == 1


async def test_a_failing_store_never_fails_the_turn():
    manager = SessionManager()
    manager.attach_store(MemoryStore(fail=True))
    first_turn(manager)
    await manager.flush()
    assert isinstance(manager.resolve("r1", "key-a"), Resolved)
    assert manager.stats().unpersisted == 1


# --- clients that name the session rather than the turn ------------------------------------------------


RESENT = (USER, {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]})
NEXT = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "and again"}]}


def _named_turn(manager: SessionManager, owner: str = "key-a", session: str = "thread-1"):
    key = client_session_key(owner, session)
    return manager.commit(
        owner=owner,
        prior=None,
        turn_input=(USER,),
        output=(ASSISTANT,),
        response_id="r1",
        now=1.0,
        new_session_id=key,
        client_session_id=session,
    )


async def test_a_resent_history_continues_the_session_it_reproduces():
    manager = SessionManager()
    first = _named_turn(manager)

    continuation = await manager.continue_client_session("key-a", "thread-1", (*RESENT, NEXT))

    assert continuation.session_id == first.id
    assert continuation.prior is not None and continuation.prior.session.id == first.id
    assert continuation.turn_input == (NEXT,)


async def test_the_first_turn_of_a_named_session_lands_where_the_next_will_look():
    manager = SessionManager()

    continuation = await manager.continue_client_session("key-a", "thread-1", (USER,))

    assert continuation.prior is None
    assert continuation.session_id == client_session_key("key-a", "thread-1")
    assert continuation.turn_input == (USER,)


async def test_input_that_does_not_reproduce_the_stored_history_starts_its_own_session():
    manager = SessionManager()
    first = _named_turn(manager)
    elsewhere = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "different"}]}

    continuation = await manager.continue_client_session("key-a", "thread-1", (elsewhere, NEXT))

    assert continuation.prior is None and continuation.session_id != first.id


async def test_a_resend_with_nothing_new_is_not_a_continuation():
    """Two identical turns must not append an empty one to the transcript."""
    manager = SessionManager()
    _named_turn(manager)

    continuation = await manager.continue_client_session("key-a", "thread-1", RESENT)

    assert continuation.prior is None


async def test_one_key_cannot_reach_another_keys_session_by_naming_it():
    manager = SessionManager()
    _named_turn(manager, owner="key-a")

    continuation = await manager.continue_client_session("key-b", "thread-1", (*RESENT, NEXT))

    assert continuation.prior is None
    assert continuation.session_id == client_session_key("key-b", "thread-1")


async def test_a_named_session_is_found_again_after_a_restart():
    store = MemoryStore()
    manager = SessionManager()
    manager.attach_store(store)
    first = _named_turn(manager)
    await manager.flush()

    restarted = SessionManager()
    restarted.attach_store(store)
    continuation = await restarted.continue_client_session("key-a", "thread-1", (*RESENT, NEXT))

    assert continuation.session_id == first.id
    assert continuation.prior is not None and continuation.turn_input == (NEXT,)
    assert restarted.stats().rehydrations == 1


# --- several conversations under one client session id ----------------------------------------------------


def _user(text: str):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _reply(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


async def _begin(manager: SessionManager, history: tuple, text: str, session: str = "thread-1"):
    sent = (*history, _user(text))
    return sent, await manager.continue_client_session("key-a", session, sent)


def _finish(manager: SessionManager, continuation, text: str, response_id: str, session: str = "thread-1"):
    return manager.commit(
        owner="key-a",
        prior=continuation.prior,
        turn_input=continuation.turn_input,
        output=(_reply(text),),
        response_id=response_id,
        now=1.0,
        new_session_id=continuation.session_id,
        client_session_id=session,
    )


async def _client_turn(manager: SessionManager, history: tuple, text: str, response_id: str) -> tuple:
    """One turn of a client that resends its whole history; the history it will send next"""
    sent, continuation = await _begin(manager, history, text)
    _finish(manager, continuation, text, response_id)
    return (*sent, _reply(text))


def _rows(manager: SessionManager, session: str = "thread-1"):
    return sorted((s for s in manager._sessions.values() if s.client_session_id == session), key=lambda s: s.id)


async def test_interleaved_conversations_under_one_client_id_are_each_stored_once():
    manager = SessionManager()
    main: tuple = ()
    subagent: tuple = ()
    for turn in range(4):
        main = await _client_turn(manager, main, f"main {turn}", f"m{turn}")
        subagent = await _client_turn(manager, subagent, f"subagent {turn}", f"s{turn}")

    rows = _rows(manager)
    assert len(rows) == 2
    assert sorted(len(row.items) for row in rows) == [8, 8]
    assert sum(len(row.items) for row in rows) == len(main) + len(subagent)
    assert manager.stats().forks == 0


async def test_two_first_turns_in_flight_never_overwrite_each_other():
    """A title request and the conversation's first turn both find the derived key free; both must be kept."""
    manager = SessionManager()
    title_sent, title = await _begin(manager, (), "name this")
    main_sent, main = await _begin(manager, (), "hello")
    assert title.session_id == main.session_id == client_session_key("key-a", "thread-1")

    _finish(manager, main, "hello", "m0")
    _finish(manager, title, "name this", "t0")

    rows = _rows(manager)
    assert len(rows) == 2
    assert sorted(len(row.items) for row in rows) == [2, 2]
    main_row = manager.resolve("m0", "key-a")
    assert isinstance(main_row, Resolved) and main_row.session.items == (*main_sent, _reply("hello"))

    after, continuation = await _begin(manager, (*main_sent, _reply("hello")), "and then")
    assert continuation.session_id == main_row.session.id
    assert continuation.turn_input == (_user("and then"),)


async def test_a_conversation_keeps_its_own_row_once_another_has_the_key():
    """The cascade seen in production: once the key held another stream, every turn forked a full copy."""
    manager = SessionManager()
    await _client_turn(manager, (), "name this", "t0")
    main: tuple = ()
    for turn in range(5):
        main = await _client_turn(manager, main, f"main {turn}", f"m{turn}")

    rows = _rows(manager)
    assert len(rows) == 2
    assert sorted(len(row.items) for row in rows) == [2, 10]


async def test_every_conversation_is_found_again_after_a_restart():
    store = MemoryStore()
    manager = SessionManager()
    manager.attach_store(store)
    main = await _client_turn(manager, (), "main 0", "m0")
    subagent = await _client_turn(manager, (), "subagent 0", "s0")
    main = await _client_turn(manager, main, "main 1", "m1")
    await manager.flush()

    restarted = SessionManager()
    restarted.attach_store(store)
    main = await _client_turn(restarted, main, "main 2", "m2")
    subagent = await _client_turn(restarted, subagent, "subagent 1", "s1")
    await restarted.flush()

    rows = _rows(restarted)
    assert sorted(len(row.items) for row in rows) == [4, 6]
    assert {row.id for row in rows} == {s.id for s in store.sessions.values()}


async def test_retrying_an_earlier_turn_forks_from_it_and_leaves_the_conversation_alone():
    manager = SessionManager()
    first = await _client_turn(manager, (), "one", "r1")
    second = await _client_turn(manager, first, "two", "r2")

    sent, retry = await _begin(manager, first, "two, differently")

    assert retry.prior is not None and retry.prior.response_id == "r1"
    assert retry.turn_input == (_user("two, differently"),)
    forked = _finish(manager, retry, "two, differently", "r2b")
    assert forked.id != retry.prior.session.id
    original = manager.resolve("r2", "key-a")
    assert isinstance(original, Resolved) and original.session.items == second


# --- a reply waits for its turn to be stored --------------------------------------------------------------


class GatedStore(MemoryStore):
    def __init__(self, fail: bool = False) -> None:
        super().__init__(fail)
        self.gate = asyncio.Event()

    async def save(self, session, first_new, response_id):
        await self.gate.wait()
        await super().save(session, first_new, response_id)


async def test_persisted_reports_the_write_once_the_store_has_it():
    store = GatedStore()
    manager = SessionManager()
    manager.attach_store(store)
    first_turn(manager)

    assert await manager.persisted("r1", timeout=0.05) is False
    store.gate.set()
    assert await manager.persisted("r1", timeout=1) is True
    assert store.saved == [(manager.resolve("r1", "key-a").session.id, 0, "r1")]


async def test_persisted_is_false_when_the_store_refuses_the_turn():
    manager = SessionManager()
    manager.attach_store(MemoryStore(fail=True))
    first_turn(manager)
    assert await manager.persisted("r1", timeout=1) is False


async def test_persisted_needs_nothing_without_a_store_and_knows_nothing_of_unqueued_turns():
    assert await SessionManager().persisted("r1", timeout=0) is True
    manager = SessionManager()
    manager.attach_store(MemoryStore())
    assert await manager.persisted("never-committed", timeout=0) is False


# --- an earlier version a compaction replaced ---------------------------------------------------------------


async def test_continuing_a_superseded_version_forks_and_leaves_the_live_session_alone():
    manager = SessionManager()
    store = MemoryStore()
    manager.attach_store(store)
    live = first_turn(manager)
    old = replace(live, id=f"{live.id}~v0", superseded=True)
    store.sessions["r1"] = old
    manager._drop(live.id)
    manager._hold(replace(live, responses=MappingProxyType({}), version=1))

    resolved = await manager.resolve_async("r1", "key-a")
    assert isinstance(resolved, Resolved) and resolved.session.superseded

    forked = manager.commit(
        owner="key-a", prior=resolved, turn_input=(CALL_OUTPUT,), output=(ASSISTANT,), response_id="r2", now=2.0
    )
    await manager.flush()
    assert forked.id not in (live.id, old.id) and not forked.superseded
    assert forked.items == (*live.items, CALL_OUTPUT, ASSISTANT)
    assert store.saved[-1] == (forked.id, 0, "r2")
    assert manager._sessions[live.id].version == 1
