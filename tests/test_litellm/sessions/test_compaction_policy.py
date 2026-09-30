import asyncio
from datetime import UTC, datetime

import pydantic
import pytest

from litellm.sessions import ReplayBase, Resolved, SessionManager, TurnUsage
from litellm.sessions.compaction import CompactionReply, compact
from litellm.sessions.compaction_policy import CompactionPolicy, PolicySettings, policy_from, policy_from_json
from litellm.sessions.compaction_presets import DEFAULT, HERMES, OPENCLAW, OPENCODE, PRESETS

DAY = datetime(2026, 9, 19, tzinfo=UTC)


def user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def assistant(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def test_every_preset_round_trips_through_its_editable_form():
    for policy in PRESETS.values():
        assert policy_from_json(policy.to_json()) == policy
        assert "{{" not in policy.request_text(False, DAY) and "{{" not in policy.request_text(True, DAY)


def test_harness_presets_carry_their_own_framing():
    items, at = OPENCODE.summary_items("S")
    assert [item["role"] for item in items] == ["user", "assistant"] and at == 1
    assert items[0]["content"][0]["text"] == "What did we do so far?"
    assert items[1]["content"][0] == {"type": "output_text", "text": "S"}

    ((claw,), _) = OPENCLAW.summary_items("S")
    assert claw["content"][0]["text"].endswith("<summary>\nS\n</summary>")
    ((hermes,), _) = HERMES.summary_items("S")
    assert hermes["content"][0]["text"].startswith("[CONTEXT COMPACTION — REFERENCE ONLY]")
    assert "2026-09-19" in HERMES.request_text(False, DAY)


def test_update_instructions_apply_only_when_the_history_starts_from_a_summary():
    assert DEFAULT.request_text(False) == DEFAULT.instructions
    assert DEFAULT.request_text(True) == DEFAULT.settings.update_instructions
    plain = CompactionPolicy("plain", "Summarize.")
    assert plain.request_text(True) == "Summarize."


def test_stored_settings_are_validated():
    assert policy_from("p", "custom", "x", '{"keep_turns": 3}').settings.keep_turns == 3
    assert policy_from("p", "custom", "x", None).settings == PolicySettings()
    with pytest.raises(pydantic.ValidationError):
        policy_from("p", "custom", "x", {"keep_turns": 0})
    with pytest.raises(pydantic.ValidationError):
        policy_from("p", "custom", "x", {"unknown": 1})
    with pytest.raises(pydantic.ValidationError):
        policy_from_json('{"name": "p"}')


REPLAY = ReplayBase("xai", "xai/grok", {})
USAGE = TurnUsage(30_000, 29_000, 100)


def session_with(manager, turns):
    prior = None
    for index, text in enumerate(turns):
        manager.commit(
            owner="k",
            prior=prior,
            turn_input=(user(text),),
            output=(assistant(f"a{index}"),),
            response_id=f"r{index}",
            now=index * 10.0 + 1,
            new_session_id="s1",
            started_at=index * 10.0,
            usage=USAGE,
            replay=REPLAY,
        )
        prior = manager.resolve(f"r{index}", "k")
        assert isinstance(prior, Resolved)
    return f"r{len(turns) - 1}"


def compact_now(manager, policy, reply_text="S", seen=None):
    (job,) = manager.due_compactions(manager.warm_until("s1") - 10, 1000, 20)

    async def call(replay, items, owner, max_output_tokens):
        if seen is not None:
            seen.append((items[-1]["content"][0]["text"], max_output_tokens))
        return CompactionReply(reply_text, False, None)

    return asyncio.run(compact(manager, job, policy, call))


def test_an_assistant_summary_is_replayed_as_model_output_and_keep_turns_is_honoured():
    manager = SessionManager()
    last = session_with(manager, ["one", "two", "three"])
    policy = CompactionPolicy(
        "two-turns", "Summarize.", PolicySettings(summary_role="assistant", summary_lead_in="Recap?", keep_turns=2)
    )
    compacted = compact_now(manager, policy)
    assert compacted is not None
    assert compacted.items == (
        user("Recap?"),
        assistant("S"),
        user("two"),
        assistant("a1"),
        user("three"),
        assistant("a2"),
    )
    assert compacted.outputs == frozenset({1, 3, 5})
    assert compacted.annotations[1]["policy"] == "two-turns"
    resolved = manager.resolve(last, "k")
    assert isinstance(resolved, Resolved) and resolved.history[1] == assistant("S")


def test_a_second_compaction_asks_for_an_update_and_never_keeps_the_old_summary_as_a_turn():
    manager = SessionManager()
    session_with(manager, ["one", "two"])
    assert compact_now(manager, DEFAULT, "first summary") is not None
    prior = manager.resolve("r1", "k")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user("three"),),
        output=(assistant("a2"),),
        response_id="r2",
        now=500.0,
        started_at=499.0,
        usage=USAGE,
        replay=REPLAY,
    )
    seen = []
    capped = CompactionPolicy(
        "capped", DEFAULT.instructions, DEFAULT.settings.model_copy(update={"max_output_tokens": 2048})
    )
    compacted = compact_now(manager, capped, "second summary", seen)
    assert seen == [(DEFAULT.settings.update_instructions, 2048)]
    assert compacted is not None and compacted.items[0]["content"][0]["text"].endswith("second summary")
    assert compacted.items[1:] == (user("three"), assistant("a2"))
