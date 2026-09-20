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
    from litellm.sessions.responses import affinity_key
    from litellm.sessions import SessionManager

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
    assert session.id != manager.resolve(_provider_id(first.id), "key-a").session.id
    assert fresh.last_turn(session.id).replay.params["prompt_cache_key"] == before["prompt_cache_key"]
    assert before["prompt_cache_key"] == affinity_key("key-a", "ses_client_2")
