from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import reduce
from typing import Final, Literal

import pytest
from pydantic import TypeAdapter

from litellm.proxy.session_import import plan_import
from litellm.proxy.sessions import PrismaTranscriptStore
from litellm.repositories.conversation_session_repository import (
    ConversationSessionRepository,
    ImportWrite,
    ItemRow,
    TranscriptConflict,
    TurnWrite,
    jsonb_text,
)
from litellm.sessions.importers.common import ImportedItem, ImportedSession

_NOW: Final = datetime(2026, 1, 2, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class Row:
    session_id: str = "session-test"
    owner: str = "owner-test"
    version: int = 0
    last_upstream_at: datetime = _NOW
    expires_at: datetime | None = None
    pinned: bool = False
    metadata: str | None = None
    source: str | None = None
    source_id: str | None = None
    created_at: datetime = _NOW
    title: str | None = None
    parent_session_id: str | None = None
    seq: int = 0
    content: str = "{}"
    is_output: bool = False
    meta: str | None = None
    search_text: str = ""
    response_id: str = ""
    item_count: int = 0


_ROW: Final = TypeAdapter(Row)


class Table:
    def __init__(self, rows: tuple[Row, ...] = ()) -> None:
        self.rows: tuple[Row, ...] = rows
        self.reads: int = 0

    async def find_unique(self, where: Mapping[str, object]) -> Row | None:
        self.reads += 1
        return next((row for row in self.rows if matches(row, where)), None)

    async def find_many(
        self, where: Mapping[str, object], order: Mapping[str, object] | None = None
    ) -> tuple[Row, ...]:
        self.reads += 1
        found: Final = tuple(row for row in self.rows if matches(row, where))
        return tuple(sorted(found, key=lambda row: row.seq)) if order == {"seq": "asc"} else found


def matches(row: Row, where: Mapping[str, object]) -> bool:
    return all(getattr(row, key) == value for key, value in where.items())


@dataclass(frozen=True, slots=True)
class Operation:
    table: Table
    kind: Literal["insert", "upsert"]
    data: Mapping[str, object]
    where: Mapping[str, object]
    skip_duplicates: bool = False


class BatchTable:
    def __init__(self, batch: Batch, table: Table) -> None:
        self.batch: Final = batch
        self.table: Final = table

    def create_many(self, data: Sequence[Mapping[str, object]], *, skip_duplicates: bool | None = None) -> None:
        self.batch.operations += tuple(
            Operation(
                self.table,
                "insert",
                row,
                {key: row[key] for key in ("session_id", "version", "seq")},
                bool(skip_duplicates),
            )
            for row in data
        )

    def upsert(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        self.batch.operations += (Operation(self.table, "upsert", data, where),)


def apply(rows: tuple[Row, ...], operation: Operation) -> tuple[Row, ...]:
    existing: Final = next((row for row in rows if matches(row, operation.where)), None)
    if operation.kind == "insert":
        if existing is None:
            return (*rows, _ROW.validate_python(operation.data))
        if operation.skip_duplicates:
            return rows
        raise RuntimeError("duplicate item sequence")
    payload: Final = operation.data["create" if existing is None else "update"]
    assert isinstance(payload, Mapping)
    updated: Final = _ROW.validate_python({**(asdict(existing) if existing is not None else {}), **payload})
    return (*tuple(row for row in rows if not matches(row, operation.where)), updated)


class Batch:
    def __init__(self, db: Database) -> None:
        self.db: Final = db
        self.operations: tuple[Operation, ...] = ()
        self.litellm_conversationsession: Final = BatchTable(self, db.litellm_conversationsession)
        self.litellm_conversationitem: Final = BatchTable(self, db.litellm_conversationitem)
        self.litellm_conversationresponse: Final = BatchTable(self, db.litellm_conversationresponse)

    async def commit(self) -> None:
        self.db.commits += 1
        if self.db.failure is not None:
            raise self.db.failure
        pending: Final = tuple((table, self._result(table)) for table in self.db.tables)
        for table, rows in pending:
            table.rows = rows

    def _result(self, table: Table) -> tuple[Row, ...]:
        return reduce(apply, (op for op in self.operations if op.table is table), table.rows)


class Database:
    def __init__(self) -> None:
        self.litellm_conversationsession: Final = Table()
        self.litellm_conversationitem: Final = Table()
        self.litellm_conversationresponse: Final = Table()
        self.tables: Final = (
            self.litellm_conversationsession,
            self.litellm_conversationitem,
            self.litellm_conversationresponse,
        )
        self.batches: int = 0
        self.commits: int = 0
        self.failure: RuntimeError | None = None

    def batch_(self) -> Batch:
        self.batches += 1
        return Batch(self)


@dataclass(frozen=True, slots=True)
class Client:
    db: Database


def turn(**changes: object) -> TurnWrite:
    base: Final = TurnWrite(
        session_id="session-test",
        owner="owner-test",
        version=2,
        last_upstream_at=_NOW,
        expires_at=_NOW + timedelta(days=1),
        items=(ItemRow(0, jsonb_text({"role": "user", "content": "synthetic question"}), False),),
        response_id="response-test",
        item_count=1,
        metadata=jsonb_text({"litellm_session_id": "client-test"}),
    )
    return TypeAdapter(TurnWrite).validate_python({**asdict(base), **changes})


def imported() -> ImportWrite:
    write: Final = turn()
    return ImportWrite(
        session_id=write.session_id,
        owner=write.owner,
        version=write.version,
        source="synthetic",
        source_id="source-test",
        title="Synthetic transcript",
        parent_session_id="parent-test",
        metadata=write.metadata or "{}",
        created_at=_NOW - timedelta(days=2),
        last_upstream_at=_NOW,
        items=write.items,
        responses=(("response-first", 1), ("response-second", 1)),
    )


async def test_save_turn_commits_session_items_and_response_together() -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    write: Final = turn()
    await repository.save_turn(write)
    stored: Final = await repository.load_by_response(write.response_id)
    assert stored is not None
    assert (stored.owner, stored.version, stored.last_upstream_at, stored.metadata) == (
        write.owner,
        write.version,
        write.last_upstream_at,
        write.metadata,
    )
    assert tuple((item.seq, item.content, item.is_output) for item in stored.items) == (
        (0, write.items[0].content, False),
    )
    assert tuple((response.response_id, response.item_count) for response in stored.responses) == (
        (write.response_id, write.item_count),
    )
    assert db.litellm_conversationsession.rows[0].expires_at == write.expires_at
    assert (db.batches, db.commits) == (1, 1)


async def test_duplicate_turn_fails_without_replacing_response_or_partial_session_update() -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    await repository.save_turn(turn())
    before: Final = tuple(table.rows for table in db.tables)
    with pytest.raises(RuntimeError, match="duplicate item sequence"):
        await repository.save_turn(turn(owner="other-owner", response_id="response-collision"))
    assert tuple(table.rows for table in db.tables) == before
    assert await repository.load_by_response("response-collision") is None


async def test_stale_turn_is_rejected_before_starting_any_batch() -> None:
    db: Final = Database()
    db.litellm_conversationsession.rows = (Row(version=3),)
    repository: Final = ConversationSessionRepository(Client(db))
    with pytest.raises(TranscriptConflict):
        await repository.save_turn(turn(version=2))
    assert (db.batches, db.commits) == (0, 0)
    assert db.litellm_conversationsession.rows[0].version == 3
    assert db.litellm_conversationitem.rows == db.litellm_conversationresponse.rows == ()


@pytest.mark.parametrize("method", ("save_turn", "import_session"))
async def test_commit_failure_propagates_without_a_loadable_transcript(method: str) -> None:
    db: Final = Database()
    failure: Final = RuntimeError("synthetic database unavailable")
    db.failure = failure
    repository: Final = ConversationSessionRepository(Client(db))
    operation: Final = repository.save_turn(turn()) if method == "save_turn" else repository.import_session(imported())
    with pytest.raises(RuntimeError) as raised:
        await operation
    assert raised.value is failure
    assert await repository.load_session("session-test") is None
    assert await repository.load_by_response("response-test") is None
    assert all(table.rows == () for table in db.tables)


@pytest.mark.parametrize(
    ("item", "expected"),
    (
        ({"role": "user", "content": "alpha\x00beta"}, "alpha\ufffdbeta"),
        (
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "part one"}, {"type": "output_text", "text": "part two"}],
            },
            "part one\npart two",
        ),
        ({"type": "function_call", "call_id": "call-test", "name": "lookup", "arguments": '{"n":1}'}, '{"n":1}'),
        ({"type": "function_call_output", "call_id": "call-test", "output": "tool result"}, "tool result"),
        ({"type": "reasoning", "summary": []}, ""),
    ),
)
async def test_persisted_search_text_is_derived_from_sanitized_content(
    item: Mapping[str, object], expected: str
) -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    content: Final = jsonb_text(item)
    await repository.save_turn(turn(items=(ItemRow(0, content, False),)))
    row: Final = db.litellm_conversationitem.rows[0]
    assert row.search_text == expected
    assert "\x00" not in row.search_text
    assert "\x00" not in str(json.loads(row.content))
    assert row.content == content


def test_jsonb_sanitizes_nested_keys_values_and_tuples_without_changing_input() -> None:
    value: Final = {"key\x00": ({"text": "a\x00b"}, None, 4, True)}
    encoded: Final = jsonb_text(value)
    assert json.loads(encoded) == {"key\ufffd": [{"text": "a\ufffdb"}, None, 4, True]}
    assert value["key\x00"][0] == {"text": "a\x00b"}


async def test_import_is_pinned_idempotent_and_preserves_source_on_update() -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    write: Final = imported()
    await repository.import_session(write)
    await repository.import_session(replace(write, source="changed-source", created_at=_NOW, title="Updated title"))
    row: Final = db.litellm_conversationsession.rows[0]
    assert (row.pinned, row.expires_at, row.owner) == (True, None, write.owner)
    assert (row.source, row.source_id, row.created_at) == (write.source, write.source_id, write.created_at)
    assert (row.title, row.parent_session_id, row.metadata) == (
        "Updated title",
        write.parent_session_id,
        write.metadata,
    )
    assert len(db.litellm_conversationitem.rows) == len(write.items)
    assert {(row.response_id, row.item_count) for row in db.litellm_conversationresponse.rows} == set(write.responses)
    assert (db.batches, db.commits) == (2, 2)


@pytest.mark.parametrize("session_version", (None, 1))
async def test_response_cannot_load_missing_or_older_version_session(session_version: int | None) -> None:
    db: Final = Database()
    db.litellm_conversationresponse.rows = (Row(response_id="response-test", version=2),)
    db.litellm_conversationsession.rows = () if session_version is None else (Row(version=session_version),)
    repository: Final = ConversationSessionRepository(Client(db))
    assert await repository.load_by_response("response-test") is None
    assert db.litellm_conversationitem.reads == db.litellm_conversationresponse.reads - 1 == 0


async def test_missing_session_and_response_return_none() -> None:
    repository: Final = ConversationSessionRepository(Client(Database()))
    assert await repository.load_by_response("absent") is None
    assert await repository.load_session("absent") is None
    assert await repository.current_transcript("absent") is None


async def test_replay_orders_current_version_items_and_keeps_outputs_metadata_and_turn_boundaries() -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    first: Final = ItemRow(0, jsonb_text({"role": "user", "content": "question"}), False)
    second: Final = ItemRow(
        1, jsonb_text({"role": "assistant", "content": "answer"}), True, jsonb_text({"origin": "synthetic"})
    )
    third: Final = ItemRow(2, jsonb_text({"role": "user", "content": "followup"}), False)
    await repository.save_turn(turn(items=(second, first), item_count=2, response_id="response-first"))
    await repository.save_turn(turn(items=(third,), item_count=3, response_id="response-second", metadata=None))
    db.litellm_conversationitem.rows += (Row(version=1, seq=0, content="{}"), Row(session_id="other", seq=0))
    db.litellm_conversationresponse.rows += (Row(version=1, response_id="response-old"),)
    transcript: Final = await repository.current_transcript("session-test")
    assert transcript is not None
    assert (transcript.owner, transcript.version, transcript.contents) == (
        "owner-test",
        2,
        (first.content, second.content, third.content),
    )
    store: Final = PrismaTranscriptStore(repository, retention=None)
    loaded: Final = await store.load("response-first")
    assert loaded is not None
    assert loaded.items == tuple(json.loads(item.content) for item in (first, second, third))
    assert loaded.outputs == frozenset({1})
    assert dict(loaded.annotations) == {1: {"origin": "synthetic"}}
    assert dict(loaded.responses) == {"response-first": 2, "response-second": 3}
    assert loaded.client_session_id == "client-test"
    assert (loaded.owner, loaded.version, loaded.last_upstream_at) == ("owner-test", 2, _NOW.timestamp())
    assert await store.load_session("session-test") == loaded
    assert not loaded.superseded

    earlier: Final = await store.load("response-old")
    assert earlier is not None and earlier.superseded
    assert (earlier.id, earlier.version, dict(earlier.responses)) == ("session-test~v1", 1, {"response-old": 0})
    assert earlier.items == ({},)


async def test_partial_transcript_is_refused_by_replay_store() -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    await repository.save_turn(turn(items=(ItemRow(1, jsonb_text({"role": "user", "content": "gap"}), False),)))
    assert await PrismaTranscriptStore(repository, retention=None).load("response-test") is None


@pytest.mark.parametrize(
    ("initial_owner", "expected_owner"), (("live-owner", "live-owner"), ("archive:synthetic", "requested-owner"))
)
async def test_import_plans_append_or_version_rewritten_history_without_stealing_live_ownership(
    initial_owner: str, expected_owner: str
) -> None:
    db: Final = Database()
    repository: Final = ConversationSessionRepository(Client(db))
    first: Final = ImportedItem({"role": "user", "content": "synthetic first"}, False, {})
    second: Final = ImportedItem({"role": "assistant", "content": "synthetic second"}, True, {})
    source: Final = ImportedSession(
        source="synthetic",
        source_id="planning-test",
        title=None,
        parent_source_id=None,
        created_at=_NOW,
        updated_at=_NOW,
        items=(first,),
        turn_ends=(1,),
        metadata={},
        skipped=0,
    )
    initial: Final = plan_import(source, initial_owner, None)
    await repository.import_session(initial.write)
    stored: Final = await repository.current_transcript(initial.write.session_id)
    assert stored is not None
    appended: Final = plan_import(replace(source, items=(first, second), turn_ends=(1, 2)), "requested-owner", stored)
    assert appended.write.owner == expected_owner
    assert appended.write.version == stored.version
    assert tuple(item.seq for item in appended.write.items) == (1,)
    await repository.import_session(appended.write)
    grown: Final = await repository.current_transcript(initial.write.session_id)
    assert grown is not None
    assert grown.owner == expected_owner
    assert tuple(json.loads(content) for content in grown.contents) == (first.item, second.item)
    unchanged: Final = plan_import(replace(source, items=(first, second), turn_ends=(1, 2)), expected_owner, grown)
    assert unchanged.action == "unchanged"
    assert unchanged.write.items == ()
    rewritten: Final = plan_import(replace(source, items=(second,)), "requested-owner", grown)
    assert rewritten.write.version == grown.version + 1
    assert rewritten.write.owner == expected_owner
    await repository.import_session(rewritten.write)
    current: Final = await repository.current_transcript(initial.write.session_id)
    assert current is not None
    assert current.version == rewritten.write.version
    assert tuple(json.loads(content) for content in current.contents) == (second.item,)
    assert len(db.litellm_conversationitem.rows) == 3
    assert db.litellm_conversationsession.rows[0].pinned
