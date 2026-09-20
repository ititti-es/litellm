import json
from datetime import UTC, datetime

from litellm.proxy.session_import import imported_response_id, imported_session_id, plan_import
from litellm.repositories.conversation_session_repository import StoredTranscript
from litellm.sessions.importers.common import ImportedItem, ImportedSession


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
