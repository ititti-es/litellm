import asyncio
import json

import pytest

from litellm.llms.claude_cli.sync import SyncRequest, set_transcript_sync, sync_transcript
from litellm.proxy.claude_cli_sync import attach_claude_cli_sync, drain_claude_cli_sync, sync_session
from litellm.proxy.session_import import imported_session_id
from litellm.repositories.conversation_session_repository import StoredTranscript

SESSION = "4f0af742-dddb-4b02-95f1-ae56834c6cb2"


class _Repository:
    """Stands in for the transcript store, recording what it was asked to write."""

    def __init__(self):
        self.writes = []
        self.stored = None

    async def current_transcript(self, session_id):
        return self.stored

    async def import_session(self, write):
        self.writes.append(write)
        self.stored = StoredTranscript(
            version=write.version, contents=tuple(json.loads(row.content) for row in write.items)
        )


def _transcript(tmp_path, *exchanges, session=SESSION):
    """A Claude Code transcript of `exchanges`, each a (prompt, reply) pair."""
    directory = tmp_path / "-home-sett-work"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    parent = None
    for index, (prompt, reply) in enumerate(exchanges):
        records.append(
            {
                "uuid": f"u{index}",
                "parentUuid": parent,
                "type": "user",
                "timestamp": "2026-09-20T10:00:00.000Z",
                "message": {"role": "user", "content": prompt},
            }
        )
        records.append(
            {
                "uuid": f"a{index}",
                "parentUuid": f"u{index}",
                "type": "assistant",
                "timestamp": "2026-09-20T10:00:01.000Z",
                "message": {
                    "id": f"msg_{index}",
                    "role": "assistant",
                    "model": "claude-opus-5",
                    "content": [{"type": "text", "text": reply}],
                },
            }
        )
        parent = f"a{index}"
    path = directory / f"{session}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _request(tmp_path, **overrides):
    return SyncRequest(
        session_id=SESSION,
        owner="hashed-key",
        client_session_id="s-1",
        projects_dir=str(tmp_path),
        **overrides,
    )


@pytest.fixture(autouse=True)
def _clear_registry():
    yield
    set_transcript_sync(None)


# --- what gets stored --------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_whole_conversation_is_stored_not_just_what_passed_through(tmp_path):
    """The proxy only ever saw the last prompt and the last reply; Claude Code has the rest."""
    _transcript(tmp_path, ("first", "one"), ("second", "two"))
    repository = _Repository()

    await sync_session(_request(tmp_path), repository)

    written = repository.writes[0]
    assert [json.loads(row.content)["content"][0]["text"] for row in written.items] == [
        "first",
        "one",
        "second",
        "two",
    ]
    assert written.session_id == imported_session_id("claude_code", SESSION)
    assert written.owner == "hashed-key"


@pytest.mark.asyncio
async def test_the_stored_session_carries_the_callers_own_session_id(tmp_path):
    """Which is how the session manager recognises this transcript as that conversation's."""
    _transcript(tmp_path, ("hi", "hello"))
    repository = _Repository()

    await sync_session(_request(tmp_path), repository)

    metadata = json.loads(repository.writes[0].metadata)
    assert metadata["litellm_session_id"] == "s-1"
    assert metadata["provider"] == "claude_cli"
    assert metadata["harness"] == "claude_code"


@pytest.mark.asyncio
async def test_a_later_turn_appends_rather_than_rewriting(tmp_path):
    _transcript(tmp_path, ("first", "one"))
    repository = _Repository()
    await sync_session(_request(tmp_path), repository)

    _transcript(tmp_path, ("first", "one"), ("second", "two"))
    await sync_session(_request(tmp_path), repository)

    assert [row.seq for row in repository.writes[1].items] == [2, 3]
    assert repository.writes[1].version == 0


@pytest.mark.asyncio
async def test_a_turn_that_added_nothing_writes_nothing(tmp_path):
    _transcript(tmp_path, ("first", "one"))
    repository = _Repository()

    await sync_session(_request(tmp_path), repository)
    await sync_session(_request(tmp_path), repository)

    assert len(repository.writes) == 1


# --- failing to store is never the caller's problem ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_session_claude_code_never_wrote_is_not_an_error(tmp_path):
    repository = _Repository()

    await sync_session(_request(tmp_path), repository)

    assert repository.writes == []


@pytest.mark.asyncio
async def test_a_store_that_refuses_the_write_is_swallowed(tmp_path):
    _transcript(tmp_path, ("hi", "hello"))

    class _Broken(_Repository):
        async def import_session(self, write):
            raise RuntimeError("no database today")

    broken = _Broken()

    await sync_session(_request(tmp_path), broken)

    assert broken.writes == [] and broken.stored is None


# --- how the provider reaches it -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_reported_turn_reaches_the_store(tmp_path):
    _transcript(tmp_path, ("hi", "hello"))
    repository = _Repository()
    assert attach_claude_cli_sync(repository) is True

    sync_transcript(_request(tmp_path))
    await drain_claude_cli_sync()

    assert len(repository.writes) == 1


@pytest.mark.asyncio
async def test_two_turns_at_once_do_not_import_over_each_other(tmp_path):
    _transcript(tmp_path, ("first", "one"), ("second", "two"))
    repository = _Repository()

    await asyncio.gather(sync_session(_request(tmp_path), repository), sync_session(_request(tmp_path), repository))

    # The second waits for the first, so it sees a stored history and finds nothing left to add.
    assert len(repository.writes) == 1


def test_without_a_transcript_store_nothing_is_reported():
    assert attach_claude_cli_sync(None) is False

    sync_transcript(SyncRequest(session_id=SESSION, owner="o", client_session_id=None, projects_dir=None))
