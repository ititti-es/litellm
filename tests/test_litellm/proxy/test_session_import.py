import json
from datetime import UTC, datetime
from typing import Final, cast

import pytest

from litellm.proxy.session_import import imported_response_id, imported_session_id, plan_import, run_import
from litellm.repositories.conversation_session_repository import (
    ConversationSessionRepository,
    ImportWrite,
    StoredTranscript,
)
from litellm.sessions.importers.common import (
    ImportedItem,
    ImportedSession,
    from_last_compaction,
    with_tool_outputs_capped,
)


def _session(*texts):
    return ImportedSession(
        source="opencode",
        source_id="ses_1",
        title="t",
        parent_source_id="ses_0",
        created_at=datetime(2026, 9, 19, tzinfo=UTC),
        updated_at=datetime(2026, 9, 19, tzinfo=UTC),
        items=tuple(ImportedItem({"role": "user", "content": text}, False, {"kind": "user"}) for text in texts),
        turn_ends=(len(texts),),
        metadata={"harness": "opencode"},
        skipped=0,
    )


def _stored(version, *texts):
    return StoredTranscript(version=version, contents=tuple({"content": text, "role": "user"} for text in texts))


def test_first_import_writes_everything_under_stable_ids():
    planned = plan_import(_session("a", "b"), "owner", None)
    assert planned.action == "new" and [row.seq for row in planned.write.items] == [0, 1]
    assert planned.write.session_id == imported_session_id("opencode", "ses_1")
    assert planned.write.parent_session_id == imported_session_id("opencode", "ses_0")
    assert planned.write.responses == ((imported_response_id("opencode", "ses_1", 0), 2),)


def test_reimport_appends_growth_and_versions_a_rewrite():
    assert plan_import(_session("a", "b"), "o", _stored(0, "a", "b")).action == "unchanged"
    grown = plan_import(_session("a", "b", "c"), "o", _stored(0, "a", "b"))
    assert grown.action == "appended" and [row.seq for row in grown.write.items] == [2]
    rewritten = plan_import(_session("a", "x"), "o", _stored(0, "a", "b"))
    assert rewritten.action == "new version 1" and rewritten.write.version == 1 and len(rewritten.write.items) == 2


def test_nul_is_stored_as_the_replacement_character_and_still_compares_equal():
    planned = plan_import(_session("a\x00b"), "o", None)
    assert json.loads(planned.write.items[0].content)["content"] == "a�b"
    assert plan_import(_session("a\x00b"), "o", _stored(0, "a�b")).action == "unchanged"


def test_an_import_does_not_take_a_session_from_the_key_using_it():
    """The claude_cli provider stores a live conversation; archiving it later must not forbid its caller."""
    live = StoredTranscript(version=0, contents=(), owner="hashed-live-key")

    assert plan_import(_session("a"), "hashed-archive-key", live).write.owner == "hashed-live-key"
    assert "kept owner" in plan_import(_session("a"), "hashed-archive-key", live).action


def test_an_archive_only_session_can_be_adopted_by_a_key():
    archived = StoredTranscript(version=0, contents=(), owner="archive:claude-code")

    planned = plan_import(_session("a"), "hashed-key", archived)

    assert planned.write.owner == "hashed-key" and "kept owner" not in planned.action


def test_a_session_nobody_owns_yet_goes_to_the_importer():
    assert plan_import(_session("a"), "hashed-key", None).write.owner == "hashed-key"


class _ImportRepository:
    def __init__(self, stored: StoredTranscript | None) -> None:
        self.stored = stored
        self.reads: tuple[str, ...] = ()
        self.writes: tuple[ImportWrite, ...] = ()

    async def current_transcript(self, session_id: str) -> StoredTranscript | None:
        self.reads += (session_id,)
        return self.stored

    async def import_session(self, write: ImportWrite) -> None:
        self.writes += (write,)


@pytest.mark.asyncio
@pytest.mark.parametrize("texts,action", [(("a", "b", "c"), "appended"), (("a", "x"), "new version 4")])
async def test_dry_run_reads_existing_history_and_reports_changes_without_writes(
    texts: tuple[str, ...], action: str
) -> None:
    repository: Final = _ImportRepository(_stored(3, "a", "b"))
    lines: Final = tuple(
        await run_import((_session(*texts),), "owner", cast(ConversationSessionRepository, repository), write=False)
    )

    assert repository.reads == (imported_session_id("opencode", "ses_1"),)
    assert repository.writes == ()
    assert len(lines) == 1 and action in lines[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["owner", "another-owner"])
async def test_unchanged_reimport_never_writes_even_when_preserving_another_owner(owner: str) -> None:
    repository: Final = _ImportRepository(StoredTranscript(version=3, contents=_stored(3, "a").contents, owner="owner"))
    lines: Final = tuple(await run_import((_session("a"),), owner, cast(ConversationSessionRepository, repository)))

    assert repository.reads == (imported_session_id("opencode", "ses_1"),)
    assert len(lines) == 1 and "unchanged" in lines[0]
    assert repository.writes == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [("a", "b", "c"), ("a", "x")])
@pytest.mark.parametrize("stored_owner", ["live-owner", "archive:opencode"])
async def test_import_executes_the_planned_append_or_rewrite_and_preserves_live_ownership(
    texts: tuple[str, ...], stored_owner: str
) -> None:
    stored: Final = StoredTranscript(version=3, contents=_stored(3, "a", "b").contents, owner=stored_owner)
    repository: Final = _ImportRepository(stored)
    session: Final = _session(*texts)
    planned: Final = plan_import(session, "importer", stored)

    lines: Final = tuple(await run_import((session,), "importer", cast(ConversationSessionRepository, repository)))

    assert repository.reads == (planned.write.session_id,)
    assert repository.writes == (planned.write,)
    assert planned.write.owner == ("live-owner" if stored_owner == "live-owner" else "importer")
    assert tuple(row.seq for row in planned.write.items) == ((2,) if texts[-1] == "c" else (0, 1))
    assert planned.write.version == (3 if texts[-1] == "c" else 4)
    assert planned.action in lines[0]


def _compacted_session():
    kinds = ("user", "text", "compaction_request", "compaction_summary", "user", "text", "user", "text")
    return ImportedSession(
        source="opencode",
        source_id="ses_1",
        title="t",
        parent_source_id=None,
        created_at=datetime(2026, 9, 19, tzinfo=UTC),
        updated_at=datetime(2026, 9, 19, tzinfo=UTC),
        items=tuple(
            ImportedItem({"role": "user", "content": str(seq)}, False, {"kind": kind}) for seq, kind in enumerate(kinds)
        ),
        turn_ends=(2, 4, 6, 8),
        metadata={"harness": "opencode"},
        skipped=0,
    )


def test_a_continued_session_starts_at_its_last_compaction_and_keeps_response_ids():
    full = _compacted_session()
    continued = from_last_compaction(full)
    assert [item.item["content"] for item in continued.items] == ["2", "3", "4", "5", "6", "7"]
    assert continued.turn_ends == (2, 4, 6) and continued.first_turn == 1
    planned = plan_import(continued, "o", _stored(0, *(str(seq) for seq in range(8))))
    assert planned.action == "new version 1"
    assert planned.write.responses == tuple(
        (imported_response_id("opencode", "ses_1", turn), end) for turn, end in ((1, 2), (2, 4), (3, 6))
    )
    assert planned.write.responses[-1][0] == plan_import(full, "o", None).write.responses[-1][0]


def test_a_session_without_a_compaction_is_left_whole():
    session = _session("a", "b")
    assert from_last_compaction(session) is session


def test_capping_cuts_only_long_tool_outputs_and_says_so():
    session = _session("a")
    long_output = ImportedItem({"type": "function_call_output", "call_id": "c", "output": "x" * 50}, False, {})
    short_output = ImportedItem({"type": "function_call_output", "call_id": "d", "output": "ok"}, False, {})
    capped = with_tool_outputs_capped(
        ImportedSession(
            **{**{f: getattr(session, f) for f in session.__slots__}, "items": (long_output, short_output)}
        ),
        10,
    )
    assert capped.items[0].item["output"] == "x" * 10 + "\n[import: output cut to its first 10 of 50 characters]"
    assert capped.items[0].item["call_id"] == "c" and capped.items[1] is short_output
