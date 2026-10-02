import asyncio
from typing import Final

import pytest

import litellm
from litellm.sessions import Forbidden, Resolved, SessionManager
from litellm.sessions.responses import PendingTurn, begin_turn
from litellm.types.llms.openai import ResponsesAPIResponse


def completed(response_id: str, text: str) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id=response_id,
        created_at=0,
        output=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
    )


async def test_a_continued_turn_sends_the_held_history_followed_by_only_the_new_input():
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))

    second = await begin_turn(manager, "resp_1", "two", "key-a")
    assert isinstance(second, PendingTurn) and isinstance(second.prior, Resolved)
    assert [item["content"][0]["text"] for item in second.upstream_input()] == ["one", "a1", "two"]


async def test_foreign_and_unknown_ids_are_distinguished():
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))
    assert await begin_turn(manager, "resp_1", "x", "key-b") == Forbidden("resp_1")
    assert await begin_turn(manager, "resp_unknown", "x", "key-a") is None


@pytest.fixture
def manager():
    held = SessionManager()
    litellm.session_manager = held
    yield held
    litellm.session_manager = None


async def test_aresponses_commits_each_turn_and_rejects_other_keys(manager):
    response = await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", mock_response="yo", litellm_metadata={"user_api_key_hash": "key-a"}
    )
    assert manager.stats().commits == 1

    with pytest.raises(litellm.BadRequestError):
        await litellm.aresponses(
            model="openai/gpt-4.1",
            input="steal",
            previous_response_id=response.id,
            mock_response="no",
            litellm_metadata={"user_api_key_hash": "key-b"},
        )
    assert manager.stats().commits == 1


async def test_aresponses_keeps_nothing_when_the_client_asks_not_to_store(manager):
    await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", store=False, mock_response="yo", litellm_metadata={"user_api_key_hash": "k"}
    )
    assert manager.stats().commits == 0


def test_cache_affinity_keys_openai_and_xai_but_never_overrides_the_client():
    from litellm.sessions.responses import cache_affinity

    assert cache_affinity("xai", "s1", {}) == {"prompt_cache_key": "s1"}
    assert cache_affinity("openai", "s1", {}) == {"prompt_cache_key": "s1"}
    assert cache_affinity("anthropic", "s1", {}) == {}
    assert cache_affinity("xai", "s1", {"prompt_cache_key": "mine"}) == {}


async def test_calls_made_by_a_session_aware_caller_are_not_tracked_twice(manager):
    from litellm.sessions.responses import MANAGED_BY_CALLER

    await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response="yo",
        litellm_metadata={"user_api_key_hash": "k", MANAGED_BY_CALLER: True},
    )
    assert manager.stats().commits == 0


async def test_aresponses_records_what_a_compaction_needs_to_reuse_the_turns_prefix(manager):
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        instructions="be terse",
        max_output_tokens=50,
        mock_response="yo",
        litellm_metadata={"user_api_key_hash": "key-a"},
        model_info={"id": "deployment-1"},
    )
    resolved = manager.resolve(_provider_id(response.id), "key-a")
    assert isinstance(resolved, Resolved)
    record = manager.last_turn(resolved.session.id)
    assert record is not None and record.replay is not None
    assert record.replay.provider == "openai" and record.replay.deployment == "deployment-1"
    assert record.replay.params["instructions"] == "be terse"
    assert record.replay.params["prompt_cache_key"] == resolved.session.id
    assert "max_output_tokens" not in record.replay.params
    assert record.usage is not None and record.started_at > 0


def _provider_id(response_id: str) -> str:
    from litellm.sessions.responses import provider_response_id

    return provider_response_id(response_id)


async def test_a_client_can_name_its_compaction_policy_in_request_metadata(manager):
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="hi",
        mock_response="yo",
        litellm_metadata={"user_api_key_hash": "key-a", "requester_metadata": {"compaction_policy": "opencode"}},
    )
    resolved = manager.resolve(_provider_id(response.id), "key-a")
    assert isinstance(resolved, Resolved)
    record = manager.last_turn(resolved.session.id)
    assert record is not None and record.replay is not None and record.replay.compaction_policy == "opencode"


def test_the_cache_key_is_the_clients_own_session_id_namespaced_per_key():
    from litellm.sessions.responses import affinity_key

    mine = affinity_key("hash-a", "ses_01ABC")
    assert mine == affinity_key("hash-a", "ses_01ABC")
    assert mine != affinity_key("hash-b", "ses_01ABC")
    assert mine != affinity_key("hash-a", "ses_other")
    assert "ses_01ABC" not in mine and mine.startswith("sess_")


async def test_a_clients_session_id_keys_the_upstream_cache_and_is_recorded(manager):
    from litellm.sessions.responses import affinity_key

    async def turn(previous=None):
        return await litellm.aresponses(
            model="openai/gpt-4.1",
            input="hi",
            previous_response_id=previous,
            mock_response="yo",
            litellm_metadata={"user_api_key_hash": "key-a"},
            litellm_session_id="ses_client_1",
        )

    first = await turn()
    resolved = manager.resolve(_provider_id(first.id), "key-a")
    assert isinstance(resolved, Resolved)
    expected = affinity_key("key-a", "ses_client_1")
    record = manager.last_turn(resolved.session.id)
    assert record is not None and record.replay is not None
    assert record.replay.params["prompt_cache_key"] == expected
    assert resolved.session.client_session_id == "ses_client_1"

    second = await turn(first.id)
    continued = manager.resolve(_provider_id(second.id), "key-a")
    assert isinstance(continued, Resolved)
    assert continued.session.id == resolved.session.id
    assert manager.last_turn(continued.session.id).replay.params["prompt_cache_key"] == expected


async def test_a_restarted_gateway_keeps_the_same_upstream_cache_key(manager):
    from litellm.sessions import SessionManager
    from litellm.sessions.responses import affinity_key

    first = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="one",
        mock_response="a",
        litellm_metadata={"user_api_key_hash": "key-a"},
        litellm_session_id="ses_client_2",
    )
    before = manager.last_turn(manager.resolve(_provider_id(first.id), "key-a").session.id).replay.params

    litellm.session_manager = SessionManager()
    after_restart = await litellm.aresponses(
        model="openai/gpt-4.1",
        input="two",
        mock_response="b",
        litellm_metadata={"user_api_key_hash": "key-a"},
        litellm_session_id="ses_client_2",
    )
    fresh = litellm.session_manager
    session = fresh.resolve(_provider_id(after_restart.id), "key-a").session
    # The row a client-named session lives at is derived from the same two values as the cache key, so a
    # restart finds the conversation as well as the warm prefix rather than beginning a new one.
    assert session.id == manager.resolve(_provider_id(first.id), "key-a").session.id
    assert fresh.last_turn(session.id).replay.params["prompt_cache_key"] == before["prompt_cache_key"]
    assert before["prompt_cache_key"] == affinity_key("key-a", "ses_client_2")


# --- a client that keeps its own history -------------------------------------------------------------


def _resent(*exchanges: tuple[str, str], then: str) -> list:
    """What a harness with its own checkpointer sends: the whole conversation, every turn."""
    history = []
    for prompt, reply in exchanges:
        history.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]})
        history.append(
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_x",
                "content": [{"type": "output_text", "text": reply, "annotations": []}],
            }
        )
    return [*history, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": then}]}]


async def test_a_resent_history_continues_its_session_instead_of_starting_another():
    """LangGraph and anything else on a local checkpointer sends no previous_response_id, only a session id."""
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a", "thread-1")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))

    second = await begin_turn(manager, None, _resent(("one", "a1"), then="two"), "key-a", "thread-1")

    assert isinstance(second, PendingTurn) and isinstance(second.prior, Resolved)
    assert second.key == first.key
    assert [item["content"][0]["text"] for item in second.turn_input] == ["two"]
    assert [item["content"][0]["text"] for item in second.upstream_input()] == ["one", "a1", "two"]


async def test_resent_tool_results_continue_when_client_omits_empty_assistant_output():
    manager: Final = SessionManager()
    first: Final = await begin_turn(manager, None, "run probe", "key-a", "thread-1")
    assert isinstance(first, PendingTurn)
    call: Final = {"type": "function_call", "id": "fc_probe", "call_id": "c1", "name": "probe", "arguments": "{}"}
    first.finish(
        ResponsesAPIResponse(
            id="resp_1",
            created_at=0,
            output=[{"type": "message", "role": "assistant", "content": []}, call],
        )
    )
    second: Final = await begin_turn(
        manager,
        None,
        [
            {"role": "user", "content": "run probe"},
            call,
            {"type": "function_call_output", "call_id": "c1", "output": "marker"},
        ],
        "key-a",
        "thread-1",
    )
    assert isinstance(second, PendingTurn) and isinstance(second.prior, Resolved)
    assert second.key == first.key
    assert tuple(item.get("type") for item in second.turn_input) == ("function_call_output",)
    assert manager.stats().sessions == 1


async def test_the_whole_thread_ends_up_in_one_session():
    manager = SessionManager()
    turn = await begin_turn(manager, None, "one", "key-a", "thread-1")
    assert isinstance(turn, PendingTurn)
    turn.finish(completed("resp_1", "a1"))
    for index, (sent, reply) in enumerate(
        ((_resent(("one", "a1"), then="two"), "a2"), (_resent(("one", "a1"), ("two", "a2"), then="three"), "a3"))
    ):
        turn = await begin_turn(manager, None, sent, "key-a", "thread-1")
        assert isinstance(turn, PendingTurn)
        turn.finish(completed(f"resp_{index + 2}", reply))

    assert manager.stats().sessions == 1
    resolved = manager.resolve("resp_3", "key-a")
    assert isinstance(resolved, Resolved)
    assert [item["content"][0]["text"] for item in resolved.session.items] == ["one", "a1", "two", "a2", "three", "a3"]


async def test_a_session_id_alone_never_reaches_another_keys_history():
    """The id is client-chosen and guessable; the owner is part of what it derives from."""
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a", "thread-1")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))

    theirs = await begin_turn(manager, None, _resent(("one", "a1"), then="two"), "key-b", "thread-1")

    assert isinstance(theirs, PendingTurn) and theirs.prior is None and theirs.key != first.key


async def test_a_different_conversation_under_a_reused_id_does_not_overwrite_the_stored_one():
    """Only input that reproduces the stored history may extend it; anything else is its own conversation."""
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a", "thread-1")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))

    other = await begin_turn(manager, None, _resent(("something else", "b1"), then="two"), "key-a", "thread-1")

    assert isinstance(other, PendingTurn) and other.prior is None and other.key != first.key
    resolved = manager.resolve("resp_1", "key-a")
    assert isinstance(resolved, Resolved) and len(resolved.session.items) == 2


async def test_a_client_that_names_its_session_and_its_turn_is_unaffected():
    """previous_response_id still decides; the session id only says which cache and which row."""
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a", "thread-1")
    assert isinstance(first, PendingTurn)
    first.finish(completed("resp_1", "a1"))

    second = await begin_turn(manager, "resp_1", "two", "key-a", "thread-1")

    assert isinstance(second, PendingTurn) and isinstance(second.prior, Resolved)
    assert [item["content"][0]["text"] for item in second.turn_input] == ["two"]


async def test_an_unnamed_session_still_gets_a_fresh_one_every_turn():
    """Without a session id there is nothing to continue from, so nothing changes for those callers."""
    manager = SessionManager()
    first = await begin_turn(manager, None, "one", "key-a")
    second = await begin_turn(manager, None, "two", "key-a")

    assert isinstance(first, PendingTurn) and isinstance(second, PendingTurn)
    assert first.key != second.key and second.prior is None


# --- store: false, recorded but never rewritten ------------------------------------------------------------


@pytest.fixture
def upstream_inputs(monkeypatch):
    """What each call actually sent upstream."""
    import importlib

    main = importlib.import_module("litellm.responses.main")  # `litellm.responses` is also a function

    sent = []
    original = main.responses

    def recording(*args, **kwargs):
        sent.append(kwargs.get("input"))
        return original(*args, **kwargs)

    monkeypatch.setattr(main, "responses", recording)
    return sent


def _client_history(*exchanges, then):
    """What opencode or Codex sends: the whole conversation, every turn, with store: false."""
    items = []
    for prompt, reply in exchanges:
        items.append({"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]})
        items.append({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": reply}]})
    return [*items, {"type": "message", "role": "user", "content": [{"type": "input_text", "text": then}]}]


async def test_store_false_is_recorded_when_the_gateway_is_its_owners_store(manager):
    manager.record_store_false = True

    await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", store=False, mock_response="yo", litellm_metadata={"user_api_key_hash": "k"}
    )

    assert manager.stats().commits == 1


async def test_a_store_false_client_keeps_its_own_history_and_the_gateway_only_records_it(manager, upstream_inputs):
    """The client owns this conversation's state: what it sends is what the provider gets."""
    manager.record_store_false = True
    meta = {"user_api_key_hash": "k"}

    await litellm.aresponses(
        model="openai/gpt-4.1",
        input=_client_history(then="one"),
        store=False,
        mock_response="a1",
        litellm_metadata=meta,
        litellm_session_id="thread-1",
    )
    second = _client_history(("one", "a1"), then="two")
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        input=second,
        store=False,
        mock_response="a2",
        litellm_metadata=meta,
        litellm_session_id="thread-1",
    )

    assert upstream_inputs[-1] == second
    resolved = manager.resolve(_provider_id(response.id), "k")
    assert isinstance(resolved, Resolved)
    assert manager.stats().sessions == 1
    assert [item["content"][0]["text"] for item in resolved.session.items] == ["one", "a1", "two", "a2"]


async def test_store_false_passes_previous_response_id_through_untouched(manager, upstream_inputs):
    """Even for a session the gateway holds, it does not substitute its own copy of the history."""
    manager.record_store_false = True
    first = await litellm.aresponses(
        model="openai/gpt-4.1", input="one", mock_response="a1", litellm_metadata={"user_api_key_hash": "k"}
    )

    await litellm.aresponses(
        model="openai/gpt-4.1",
        input="two",
        store=False,
        previous_response_id=first.id,
        mock_response="a2",
        litellm_metadata={"user_api_key_hash": "k"},
    )

    assert upstream_inputs[-1] == "two"


async def test_store_false_is_still_not_recorded_by_default(manager):
    assert manager.record_store_false is False

    await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", store=False, mock_response="yo", litellm_metadata={"user_api_key_hash": "k"}
    )

    assert manager.stats().commits == 0


# --- the gateway owns the history: durable before the reply, loud when it does not hold an id --------------


class _RecordingStore:
    def __init__(self) -> None:
        self.saved: list = []

    async def save(self, session, first_new, response_id):
        await asyncio.sleep(0.05)
        self.saved.append(response_id)

    async def load(self, response_id):
        return None

    async def load_session(self, session_id):
        return None

    async def client_sessions(self, owner, client_session_id, limit):
        return ()


async def test_a_reply_is_handed_back_only_once_its_turn_is_stored(manager):
    store = _RecordingStore()
    manager.attach_store(store)

    response = await litellm.aresponses(
        model="openai/gpt-4.1", input="hi", mock_response="yo", litellm_metadata={"user_api_key_hash": "k"}
    )

    assert store.saved == [_provider_id(response.id)]


async def test_an_id_the_gateway_does_not_hold_is_refused_with_a_404_naming_it(manager):
    with pytest.raises(litellm.NotFoundError) as refused:
        await litellm.aresponses(
            model="openai/gpt-4.1",
            input="two",
            previous_response_id="resp_unknown",
            mock_response="a2",
            litellm_metadata={"user_api_key_hash": "k"},
        )

    assert refused.value.status_code == 404
    assert "response id resp_unknown" in str(refused.value)
    assert "previous_response_not_found" in str(refused.value)
    assert manager.stats().commits == 0


async def test_unknown_ids_still_pass_through_when_refusal_is_turned_off(manager, upstream_inputs):
    manager.refuse_unknown_previous_response = False

    await litellm.aresponses(
        model="openai/gpt-4.1",
        input="two",
        previous_response_id="resp_unknown",
        mock_response="a2",
        litellm_metadata={"user_api_key_hash": "k"},
    )

    assert upstream_inputs[-1] == "two"


async def test_a_store_false_client_is_never_refused_for_an_id_it_owns(manager, upstream_inputs):
    manager.record_store_false = True

    await litellm.aresponses(
        model="openai/gpt-4.1",
        input="two",
        store=False,
        previous_response_id="resp_theirs",
        mock_response="a2",
        litellm_metadata={"user_api_key_hash": "k"},
    )

    assert upstream_inputs[-1] == "two"


# --- one client session id, several conversations: each stored once ---------------------------------------


async def test_interleaved_conversations_under_one_client_session_id_are_each_stored_once(manager):
    """opencode sends its subagents' turns under the parent's session id, each resending its own whole history."""
    manager.record_store_false = True
    histories: dict[str, list] = {"main": [], "subagent": []}

    async def turn(stream: str, text: str):
        histories[stream].append({"type": "message", "role": "user", "content": f"{stream}: {text}"})
        await litellm.aresponses(
            model="openai/gpt-4.1",
            input=list(histories[stream]),
            store=False,
            mock_response=f"{stream} reply to {text}",
            litellm_metadata={"user_api_key_hash": "key-a"},
            litellm_session_id="ses_parent",
        )
        histories[stream].append({"type": "message", "role": "assistant", "content": f"{stream} reply to {text}"})

    for text in ("one", "two", "three"):
        await turn("main", text)
        await turn("subagent", text)

    rows = [s for s in manager._sessions.values() if s.client_session_id == "ses_parent"]
    assert len(rows) == 2
    assert [len(s.items) for s in rows] == [6, 6]  # the mock reuses one response id, so count items
