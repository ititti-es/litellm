import pydantic
import pytest

from litellm.proxy.sessions import configured_session_manager


def test_a_reload_with_the_same_settings_keeps_the_live_sessions():
    first = configured_session_manager({"enabled": True, "budget_mb": 8}, None)
    assert first is not None and first.budget_bytes == 8 << 20
    assert configured_session_manager({"enabled": True, "budget_mb": 8}, first) is first
    resized = configured_session_manager({"enabled": True, "budget_mb": 16}, first)
    assert resized is not first and resized is not None and resized.budget_bytes == 16 << 20


def test_absent_or_disabled_settings_turn_the_manager_off():
    live = configured_session_manager({"enabled": True}, None)
    assert configured_session_manager(None, live) is None
    assert configured_session_manager({"enabled": False}, live) is None


def test_a_non_positive_budget_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        configured_session_manager({"enabled": True, "budget_mb": 0}, None)


class FakeRepository:
    def __init__(self, stored=None):
        self.writes: list = []
        self.stored = stored

    async def save_turn(self, write):
        self.writes.append(write)

    async def load_by_response(self, response_id):
        return self.stored


def _stored(seqs):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from litellm.repositories.conversation_session_repository import StoredSession

    return StoredSession(
        session_id="s1",
        owner="k",
        version=0,
        last_upstream_at=datetime(2026, 9, 19, tzinfo=UTC),
        items=tuple(
            SimpleNamespace(seq=seq, content={"role": "user", "content": str(seq)}, is_output=seq == 1) for seq in seqs
        ),
        responses=(SimpleNamespace(response_id="r1", item_count=len(seqs)),),
    )


async def test_the_store_writes_new_items_with_their_output_flags():
    import json
    from types import MappingProxyType

    from litellm.proxy.sessions import PrismaTranscriptStore
    from litellm.sessions import Session

    repository = FakeRepository()
    session = Session(
        id="s1",
        owner="k",
        items=(
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ),
        outputs=frozenset({1}),
        responses=MappingProxyType({"r1": 2, "r2": 3}),
        version=0,
        last_upstream_at=0.0,
    )
    await PrismaTranscriptStore(repository, retention=None).save(session, first_new=1, response_id="r2")

    (write,) = repository.writes
    assert [(row.seq, json.loads(row.content)["content"], row.is_output) for row in write.items] == [
        (1, "b", True),
        (2, "c", False),
    ]
    assert write.item_count == 3 and write.expires_at is None


async def test_a_transcript_with_a_missing_row_is_never_resumed():
    from litellm.proxy.sessions import PrismaTranscriptStore

    complete = await PrismaTranscriptStore(FakeRepository(_stored([0, 1, 2])), None).load("r1")
    assert complete is not None and complete.outputs == frozenset({1}) and complete.responses["r1"] == 3
    assert await PrismaTranscriptStore(FakeRepository(_stored([0, 2])), None).load("r1") is None


async def test_item_meta_is_written_and_read_back_with_the_transcript():
    import json
    from types import MappingProxyType, SimpleNamespace

    from litellm.proxy.sessions import PrismaTranscriptStore
    from litellm.sessions import Session

    repository = FakeRepository()
    session = Session(
        id="s1",
        owner="k",
        items=({"role": "user", "content": "summary"}, {"role": "user", "content": "next"}),
        outputs=frozenset(),
        responses=MappingProxyType({"r2": 2}),
        version=1,
        last_upstream_at=0.0,
        annotations=MappingProxyType({0: MappingProxyType({"kind": "compaction_summary"})}),
    )
    await PrismaTranscriptStore(repository, retention=None).save(session, first_new=0, response_id="r2")
    (write,) = repository.writes
    assert [row.meta for row in write.items] == [json.dumps({"kind": "compaction_summary"}), None]

    stored = _stored([0, 1])
    rows = (SimpleNamespace(**{**vars(stored.items[0]), "meta": write.items[0].meta}), stored.items[1])
    loaded = await PrismaTranscriptStore(
        FakeRepository(stored.__class__(**{**_fields(stored), "items": rows})), None
    ).load("r1")
    assert loaded is not None and dict(loaded.annotations) == {0: {"kind": "compaction_summary"}}


def _fields(stored):
    return {name: getattr(stored, name) for name in stored.__slots__}


class PolicyRepository:
    def __init__(self, stored=None, error=None):
        self.stored = stored
        self.error = error

    async def find_policy(self, name):
        if self.error is not None:
            raise self.error
        return self.stored


async def test_policies_resolve_stored_then_built_in_then_default():
    from types import SimpleNamespace

    from litellm.proxy.sessions import load_policy
    from litellm.sessions.compaction_presets import DEFAULT, OPENCODE

    stored = SimpleNamespace(name="mine", source="custom", instructions="Summarize tersely", settings={"keep_turns": 2})
    mine = await load_policy(PolicyRepository(stored), "mine")
    assert mine.instructions == "Summarize tersely" and mine.settings.keep_turns == 2
    assert await load_policy(PolicyRepository(None), "opencode") == OPENCODE
    assert await load_policy(PolicyRepository(None), "missing", "opencode") == OPENCODE
    assert await load_policy(PolicyRepository(None), "missing") is DEFAULT
    assert await load_policy(PolicyRepository(error=RuntimeError("db down")), "opencode") == OPENCODE
    assert await load_policy(None, None, None) is DEFAULT
    broken = SimpleNamespace(name="opencode", source="custom", instructions="x", settings={"summary_role": "system"})
    assert await load_policy(PolicyRepository(broken), "opencode") == OPENCODE


async def test_compaction_calls_reach_the_deployment_that_served_the_turn():
    from litellm.proxy.sessions import router_upstream
    from litellm.sessions import ReplayBase
    from litellm.sessions.responses import MANAGED_BY_CALLER

    sent = []

    async def send(**kwargs):
        sent.append(kwargs)
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "notes"}]}]}

    class Router:
        def get_deployment_credentials_with_provider(self, model_id):
            assert model_id == "d1"
            return {"api_key": "sk-upstream", "custom_llm_provider": "xai", "model": "xai/alias"}

    replay = ReplayBase("xai", "xai/grok", {"prompt_cache_key": "s1", "stream": True}, deployment="d1")
    reply = await router_upstream(lambda: Router(), send)(replay, [{"type": "message"}], "owner-hash", 3000)

    assert reply.text == "notes"
    (call,) = sent
    assert call["api_key"] == "sk-upstream" and call["model"] == "xai/grok" and call["stream"] is False
    assert call["prompt_cache_key"] == "s1" and call["max_output_tokens"] == 3000
    assert call["litellm_metadata"][MANAGED_BY_CALLER] is True
    assert call["litellm_metadata"]["user_api_key_hash"] == "owner-hash"


async def test_due_sessions_are_compacted_with_the_configured_policy():
    from litellm.proxy._types import SessionCompactionSettings
    from litellm.proxy.sessions import _start_due
    from litellm.sessions import ReplayBase, Resolved, SessionManager, TurnUsage
    from litellm.sessions.compaction import CompactionReply

    manager = SessionManager()
    replay = ReplayBase("xai", "xai/grok", {})
    usage = TurnUsage(30_000, 29_000, 100)
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(user,),
        output=(),
        response_id="r1",
        now=1.0,
        new_session_id="s1",
        started_at=0.0,
        usage=usage,
        replay=replay,
    )
    prior = manager.resolve("r1", "k")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user,),
        output=(),
        response_id="r2",
        now=11.0,
        started_at=10.0,
        usage=usage,
        replay=replay,
    )
    instructions = []

    async def call(replay, items, owner, max_output_tokens):
        instructions.append(items[-1]["content"][0]["text"])
        return CompactionReply("notes", False, None)

    from types import SimpleNamespace

    settings = SessionCompactionSettings(enabled=True, policy="mine")
    deadline = manager.warm_until("s1")
    assert await _start_due(manager, settings, call, PolicyRepository(), deadline - 60, lambda model: 40_000) == ()
    tasks = await _start_due(
        manager,
        settings,
        call,
        PolicyRepository(SimpleNamespace(name="mine", source="custom", instructions="Be brief", settings=None)),
        deadline - 5,
        lambda model: 40_000,
    )
    (compacted,) = [await task for task in tasks]
    assert instructions == ["Be brief"]
    assert compacted is not None and compacted.version == 1


def test_compaction_settings_are_off_unless_asked_for():
    from litellm.proxy._types import SessionManagerSettings

    settings = SessionManagerSettings.model_validate({"enabled": True})
    assert settings.compaction.enabled is False
    assert settings.compaction.min_prefix_tokens == 20_000
    with pytest.raises(pydantic.ValidationError):
        SessionManagerSettings.model_validate({"enabled": True, "compaction": {"target_hit_rate": 1.5}})


def test_thresholds_come_from_settings_then_the_model_then_the_policy():
    from litellm.proxy._types import SessionCompactionSettings
    from litellm.proxy.sessions import compaction_thresholds
    from litellm.sessions import ReplayBase
    from litellm.sessions.compaction_policy import CompactionPolicy, PolicySettings
    from litellm.sessions.compaction_presets import DEFAULT, OFF
    from litellm.sessions.compaction_thresholds import Thresholds

    settings = SessionCompactionSettings.model_validate(
        {"enabled": True, "models": {"grok": {"min_context_fraction": 0.7}, "xai/other": {"enabled": False}}}
    )
    grok = ReplayBase("xai", "xai/grok-build-0.1", {}, model_group="grok")
    assert compaction_thresholds(settings, grok, DEFAULT, 256_000) == Thresholds(
        min_prefix_tokens=20_000, min_context_fraction=0.7, context_window=256_000
    )
    other = ReplayBase("xai", "xai/other", {})
    assert not compaction_thresholds(settings, other, DEFAULT, 256_000).enabled
    assert not compaction_thresholds(settings, grok, OFF, 256_000).enabled
    eager = CompactionPolicy("eager", "Summarize.", PolicySettings(min_context_fraction=0.1))
    assert compaction_thresholds(settings, grok, eager, 256_000).min_context_fraction == 0.1


async def test_sessions_below_the_bars_keep_their_history():
    from litellm.proxy._types import SessionCompactionSettings
    from litellm.proxy.sessions import _start_due
    from litellm.sessions import ReplayBase, Resolved, SessionManager, TurnUsage

    manager = SessionManager()
    replay = ReplayBase("xai", "xai/grok-build-0.1", {}, model_group="grok")
    usage = TurnUsage(20_000, 19_000, 500)
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "q"}]}
    manager.commit(
        owner="k",
        prior=None,
        turn_input=(user,),
        output=(),
        response_id="r1",
        now=1.0,
        new_session_id="s1",
        started_at=0.0,
        usage=usage,
        replay=replay,
    )
    prior = manager.resolve("r1", "k")
    assert isinstance(prior, Resolved)
    manager.commit(
        owner="k",
        prior=prior,
        turn_input=(user,),
        output=(),
        response_id="r2",
        now=11.0,
        started_at=10.0,
        usage=usage,
        replay=replay,
    )
    called = []

    async def call(replay, items, owner, max_output_tokens):
        called.append(items)

    settings = SessionCompactionSettings(enabled=True)
    assert await _start_due(manager, settings, call, None, manager.warm_until("s1") - 5, lambda model: 256_000) == ()
    assert called == []


async def test_the_clients_session_id_is_stored_and_read_back():
    import json
    from types import MappingProxyType, SimpleNamespace

    from litellm.proxy.sessions import PrismaTranscriptStore
    from litellm.sessions import Session

    repository = FakeRepository()
    session = Session(
        id="s1",
        owner="k",
        items=({"role": "user", "content": "a"},),
        outputs=frozenset(),
        responses=MappingProxyType({"r1": 1}),
        version=0,
        last_upstream_at=0.0,
        client_session_id="ses_client_9",
    )
    await PrismaTranscriptStore(repository, retention=None).save(session, first_new=0, response_id="r1")
    (write,) = repository.writes
    assert json.loads(write.metadata) == {"litellm_session_id": "ses_client_9"}

    stored = _stored([0])
    with_metadata = stored.__class__(**{**_fields(stored), "metadata": write.metadata})
    loaded = await PrismaTranscriptStore(FakeRepository(with_metadata), None).load("r1")
    assert loaded is not None and loaded.client_session_id == "ses_client_9"


def test_record_store_false_is_off_unless_configured():
    from litellm.proxy.sessions import configured_session_manager

    assert configured_session_manager({"enabled": True}, None).record_store_false is False
    assert configured_session_manager({"enabled": True, "record_store_false": True}, None).record_store_false is True


def test_a_reload_can_change_record_store_false_without_dropping_held_sessions():
    from litellm.proxy.sessions import configured_session_manager

    live = configured_session_manager({"enabled": True}, None)
    reloaded = configured_session_manager({"enabled": True, "record_store_false": True}, live)

    assert reloaded is live and live.record_store_false is True
