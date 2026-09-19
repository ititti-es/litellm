from litellm.sessions import Forbidden, Resolved, SessionManager, Unknown

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
