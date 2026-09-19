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
        items=({"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}),
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
