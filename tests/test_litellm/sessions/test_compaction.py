import asyncio
from types import SimpleNamespace

from litellm.sessions import ReplayBase, Resolved, SessionManager, TurnUsage
from litellm.sessions.compaction import (
    DEFAULT,
    CompactionReply,
    compact,
    replay_params,
    reply_from,
    turn_usage,
)


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


REPLAY = ReplayBase(provider="xai", model="xai/grok", params={"prompt_cache_key": "s1", "tools": []}, deployment="d1")
BIG = TurnUsage(input_tokens=30_000, cached_tokens=29_000, output_tokens=500)


def two_turns(manager, first_at=0.0, second_at=100.0, usage=BIG):
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(user("one"),),
        output=(assistant("a1"),),
        response_id="r1",
        now=first_at + 5,
        new_session_id="s1",
        started_at=first_at,
        usage=usage,
        replay=REPLAY,
    )
    prior = manager.resolve("r1", "k")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user("two"),),
        output=(assistant("a2"),),
        response_id="r2",
        now=second_at + 5,
        started_at=second_at,
        usage=usage,
        replay=REPLAY,
    )


def test_each_continued_turn_teaches_the_clock_about_its_gap():
    manager = SessionManager()
    two_turns(manager)
    estimate = manager.cache_estimate("s1")
    assert estimate is not None and (estimate.samples, estimate.hits) == (1, 1)
    assert manager.warm_until("s1") == 100.0 + estimate.ttl_seconds


def test_no_observation_when_the_model_changes_between_turns():
    manager = SessionManager()
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(user("one"),),
        output=(),
        response_id="r1",
        now=1.0,
        started_at=0.0,
        usage=BIG,
        replay=REPLAY,
    )
    prior = manager.resolve("r1", "k")
    assert isinstance(prior, Resolved)
    other = ReplayBase(provider="xai", model="xai/other", params={})
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user("two"),),
        output=(),
        response_id="r2",
        now=2.0,
        started_at=1.0,
        usage=BIG,
        replay=other,
    )
    assert manager.clock.estimate("xai", "xai/grok").samples == 0
    assert manager.clock.estimate("xai", "xai/other").samples == 0


def test_a_session_is_due_only_inside_the_lead_window_before_expiry():
    manager = SessionManager()
    two_turns(manager)
    deadline = manager.warm_until("s1")
    assert deadline is not None
    assert manager.due_compactions(deadline - 30, min_prefix_tokens=1000, lead_seconds=20) == ()
    assert manager.due_compactions(deadline + 1, min_prefix_tokens=1000, lead_seconds=20) == ()
    (job,) = manager.due_compactions(deadline - 10, min_prefix_tokens=1000, lead_seconds=20)
    assert job.response_id == "r2" and job.tail_start == 2
    assert job.history == (user("one"), assistant("a1"), user("two"), assistant("a2"))
    assert manager.due_compactions(deadline - 5, min_prefix_tokens=1000, lead_seconds=20) == ()


def test_small_sessions_and_sessions_with_a_turn_in_flight_are_left_alone():
    small = SessionManager()
    two_turns(small, usage=TurnUsage(input_tokens=2000, cached_tokens=1900, output_tokens=10))
    deadline = small.warm_until("s1")
    assert deadline is not None
    assert small.due_compactions(deadline - 10, min_prefix_tokens=20_000, lead_seconds=20) == ()

    busy = SessionManager()
    two_turns(busy)
    busy.begin("s1", 150.0)
    deadline = busy.warm_until("s1")
    assert deadline is not None
    assert busy.due_compactions(deadline - 10, min_prefix_tokens=1000, lead_seconds=20) == ()


def fake_upstream(reply, calls):
    async def call(replay, items, owner, max_output_tokens):
        calls.append((replay, items, owner))
        return reply

    return call


def test_compaction_reuses_the_prefix_and_rebases_onto_summary_and_latest_exchange():
    manager = SessionManager()
    two_turns(manager)
    deadline = manager.warm_until("s1")
    (job,) = manager.due_compactions(deadline - 10, min_prefix_tokens=1000, lead_seconds=20)
    calls = []
    reply = CompactionReply("the user counted to two", False, TurnUsage(30_100, 29_900, 40))
    compacted = asyncio.run(compact(manager, job, DEFAULT, fake_upstream(reply, calls)))

    ((replay, items, owner),) = calls
    assert replay is REPLAY and owner == "k"
    assert items[:-1] == list(job.history)
    assert items[-1]["content"][0]["text"] == DEFAULT.instructions

    assert compacted is not None and compacted.version == 1
    assert compacted.items[0]["content"][0]["text"] == DEFAULT.settings.summary_prefix + "the user counted to two"
    assert compacted.items[1:] == (user("two"), assistant("a2"))
    assert compacted.outputs == frozenset({2})
    meta = compacted.annotations[0]
    assert meta["kind"] == "compaction_summary" and meta["covers"] == {"version": 0, "first_seq": 0, "last_seq": 1}
    assert meta["usage"]["cached_tokens"] == 29_900

    resolved = manager.resolve("r2", "k")
    assert isinstance(resolved, Resolved) and resolved.history == compacted.items
    assert not isinstance(manager.resolve("r1", "k"), Resolved)


def test_the_turn_after_compaction_is_not_read_as_a_cache_miss():
    manager = SessionManager()
    two_turns(manager)
    deadline = manager.warm_until("s1")
    (job,) = manager.due_compactions(deadline - 10, min_prefix_tokens=1000, lead_seconds=20)
    asyncio.run(compact(manager, job, DEFAULT, fake_upstream(CompactionReply("s", False, None), [])))
    prior = manager.resolve("r2", "k")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user("three"),),
        output=(assistant("a3"),),
        response_id="r3",
        now=900.0,
        started_at=899.0,
        usage=TurnUsage(1500, 0, 10),
        replay=REPLAY,
    )
    assert manager.clock.estimate("xai", "xai/grok").samples == 1
    later = manager.resolve("r3", "k")
    assert isinstance(later, Resolved) and later.session.annotations[0]["kind"] == "compaction_summary"


def test_a_turn_that_lands_during_compaction_wins():
    manager = SessionManager()
    two_turns(manager)
    deadline = manager.warm_until("s1")
    (job,) = manager.due_compactions(deadline - 10, min_prefix_tokens=1000, lead_seconds=20)
    prior = manager.resolve("r2", "k")
    assert isinstance(prior, Resolved)
    manager.commit(owner="k", prior=prior, turn_input=(user("three"),), output=(), response_id="r3", now=300.0)
    assert asyncio.run(compact(manager, job, DEFAULT, fake_upstream(CompactionReply("s", False, None), []))) is None
    assert manager.resolve("r3", "k").session.version == 0


def test_a_reply_that_calls_tools_or_fails_leaves_the_session_untouched():
    for call in (
        fake_upstream(CompactionReply("", True, None), []),
        fake_upstream(CompactionReply("   ", False, None), []),
    ):
        manager = SessionManager()
        two_turns(manager)
        (job,) = manager.due_compactions(manager.warm_until("s1") - 10, 1000, 20)
        assert asyncio.run(compact(manager, job, DEFAULT, call)) is None
        assert manager.resolve("r2", "k").session.version == 0

    async def broken(replay, items, owner, max_output_tokens):
        raise RuntimeError("upstream down")

    manager = SessionManager()
    two_turns(manager)
    (job,) = manager.due_compactions(manager.warm_until("s1") - 10, 1000, 20)
    assert asyncio.run(compact(manager, job, DEFAULT, broken)) is None


def test_a_session_whose_only_exchange_is_the_tail_is_not_compacted():
    manager = SessionManager()
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(user("one"),),
        output=(assistant("a1"),),
        response_id="r1",
        now=1.0,
        new_session_id="s1",
        started_at=0.0,
        usage=BIG,
        replay=REPLAY,
    )
    assert manager.due_compactions(manager.warm_until("s1") - 10, 1000, 20) == ()


def test_replies_and_usage_are_read_from_models_or_json():
    response = SimpleNamespace(
        output=[{"type": "message", "content": [{"type": "output_text", "text": "sum"}]}],
        usage={"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 8}},
    )
    assert reply_from(response) == CompactionReply("sum", False, TurnUsage(10, 8, 2))
    assert reply_from({"output": [{"type": "function_call"}]}).called_tools
    assert turn_usage(None) is None
    assert turn_usage({"input_tokens": 3}) == TurnUsage(3, 0, 0)


def test_only_prefix_shaping_parameters_are_replayed():
    params = replay_params(
        {"tools": [], "api_key": "sk-secret", "stream": True, "prompt_cache_key": "s", "top_p": None}
    )
    assert dict(params) == {"tools": [], "prompt_cache_key": "s"}


def test_a_compaction_reports_whether_the_cache_survived_its_gap():
    manager = SessionManager()
    two_turns(manager)
    (job,) = manager.due_compactions(manager.warm_until("s1") - 10, 1000, 20)
    before = manager.clock.estimate("xai", "xai/grok").samples
    reply = CompactionReply("notes", False, TurnUsage(30_100, 0, 40))
    asyncio.run(compact(manager, job, DEFAULT, fake_upstream(reply, [])))
    estimate = manager.clock.estimate("xai", "xai/grok")
    assert estimate.samples == before + 1 and estimate.hits == before
