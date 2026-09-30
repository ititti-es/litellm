import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, cast

import pytest

from litellm.proxy.memory_backfill import BackfillStats, backfill, segment_node_id, session_node_id
from litellm.sessions.session import Item

if TYPE_CHECKING:
    from prisma import Prisma


@dataclass(frozen=True, slots=True)
class _Session:
    session_id: str
    version: int


@dataclass(frozen=True, slots=True)
class _Item:
    session_id: str
    version: int
    seq: int
    content: Item
    meta: Mapping[str, object]
    search_text: str | None = None


class _Sessions:
    def __init__(self, rows: tuple[_Session, ...]) -> None:
        self.rows = rows
        self.reads = 0

    async def find_many(self) -> tuple[_Session, ...]:
        self.reads += 1
        return self.rows


class _Items:
    def __init__(self, rows: tuple[_Item, ...]) -> None:
        self.rows = rows
        self.reads: tuple[Mapping[str, object], ...] = ()
        self.updates: tuple[tuple[Mapping[str, object], Mapping[str, object]], ...] = ()

    async def find_many(self, *, where: Mapping[str, object], order: Mapping[str, object]) -> tuple[_Item, ...]:
        self.reads += (where,)
        return tuple(
            sorted(
                (row for row in self.rows if row.session_id == where["session_id"] and row.version == where["version"]),
                key=lambda row: row.seq,
                reverse=order["seq"] == "desc",
            )
        )

    async def update(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        key: Final = cast(Mapping[str, object], where["session_id_version_seq"])
        self.updates += ((where, data),)
        self.rows = tuple(
            replace(row, search_text=cast(str, data["search_text"]))
            if (row.session_id, row.version, row.seq) == (key["session_id"], key["version"], key["seq"])
            else row
            for row in self.rows
        )


class _Nodes:
    def __init__(self) -> None:
        self.rows: Mapping[str, Mapping[str, object]] = {}
        self.upserts: tuple[tuple[Mapping[str, object], Mapping[str, object]], ...] = ()
        self.updates: tuple[tuple[Mapping[str, object], Mapping[str, object]], ...] = ()

    async def upsert(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        node_id: Final = cast(str, where["node_id"])
        fields: Final = cast(Mapping[str, object], data["update" if node_id in self.rows else "create"])
        self.upserts += ((where, data),)
        self.rows = {**self.rows, node_id: {**self.rows.get(node_id, {}), **fields}}

    async def update(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        node_id: Final = cast(str, where["node_id"])
        self.updates += ((where, data),)
        self.rows = {**self.rows, node_id: {**self.rows[node_id], **data}}


class _Client:
    def __init__(self) -> None:
        self.litellm_conversationsession = _Sessions((_Session("synthetic-session", 2), _Session("plain-session", 0)))
        self.litellm_conversationitem = _Items(
            (
                _Item("synthetic-session", 2, 0, {"role": "user", "content": "searchable"}, {}),
                _Item("synthetic-session", 2, 1, {"role": "user", "content": "do not replace"}, {}, "curated"),
                _Item("synthetic-session", 2, 2, {"type": "reasoning", "content": "hidden"}, {}),
                _Item(
                    "synthetic-session",
                    2,
                    3,
                    {"role": "assistant", "content": "later"},
                    {
                        "kind": "compaction_summary",
                        "summary": "later summary",
                        "policy": "synthetic-policy",
                        "provenance": {"model": "synthetic-model"},
                        "usage": {"output_tokens": 7},
                        "covers": {"version": 2, "first_seq": 5, "last_seq": 9},
                    },
                ),
                _Item(
                    "synthetic-session",
                    2,
                    4,
                    {"role": "assistant", "content": "earlier summary"},
                    {
                        "kind": "compaction_summary",
                    },
                    "",
                ),
                _Item("synthetic-session", 1, 0, {"role": "user", "content": "old version"}, {}),
                _Item("plain-session", 0, 0, {"role": "user", "content": "plain"}, {}),
            )
        )
        self.litellm_memorysummarynode = _Nodes()


@pytest.mark.asyncio
async def test_backfill_dry_run_reads_current_items_and_counts_without_any_writes() -> None:
    client: Final = _Client()
    before: Final = client.litellm_conversationitem.rows

    stats: Final = await backfill(cast("Prisma", client), dry_run=True)

    assert stats == BackfillStats(sessions=2, items_text_filled=3, segment_nodes=2, session_nodes=1)
    assert client.litellm_conversationsession.reads == 1
    assert client.litellm_conversationitem.reads == (
        {"session_id": "synthetic-session", "version": 2},
        {"session_id": "plain-session", "version": 0},
    )
    assert client.litellm_conversationitem.rows == before
    assert client.litellm_conversationitem.updates == ()
    assert client.litellm_memorysummarynode.upserts == ()
    assert client.litellm_memorysummarynode.updates == ()
    assert client.litellm_memorysummarynode.rows == {}


@pytest.mark.asyncio
async def test_backfill_fills_only_null_text_and_upserts_linked_nodes_idempotently() -> None:
    client: Final = _Client()
    stats: Final = await backfill(cast("Prisma", client), dry_run=False)
    items: Final = client.litellm_conversationitem
    nodes: Final = client.litellm_memorysummarynode
    parent: Final = session_node_id("synthetic-session")
    later: Final = segment_node_id("synthetic-session", 2, 3)
    earlier: Final = segment_node_id("synthetic-session", 2, 4)

    assert stats == BackfillStats(sessions=2, items_text_filled=3, segment_nodes=2, session_nodes=1)
    assert tuple(row.search_text for row in items.rows) == ("searchable", "curated", "", "later", "", None, "plain")
    assert tuple(where["session_id_version_seq"] for where, _ in items.updates) == (
        {"session_id": "synthetic-session", "version": 2, "seq": 0},
        {"session_id": "synthetic-session", "version": 2, "seq": 2},
        {"session_id": "synthetic-session", "version": 2, "seq": 3},
        {"session_id": "plain-session", "version": 0, "seq": 0},
    )
    assert set(nodes.rows) == {parent, later, earlier}
    assert nodes.rows[parent] == {
        "node_id": parent,
        "session_id": "synthetic-session",
        "level": "session",
        "summary": "- earlier summary\n\n- later summary",
        "tokens": 7,
    }
    assert nodes.rows[later]["summary"] == "later summary"
    assert nodes.rows[later]["policy_id"] == "synthetic-policy"
    assert nodes.rows[later]["model"] == "synthetic-model"
    assert nodes.rows[later]["tokens"] == 7
    assert tuple(nodes.rows[later][key] for key in ("span_version", "span_first_seq", "span_last_seq")) == (2, 5, 9)
    assert nodes.rows[earlier]["summary"] == "earlier summary"
    assert tuple(nodes.rows[earlier][key] for key in ("span_version", "span_first_seq", "span_last_seq")) == (
        None,
        None,
        None,
    )
    assert json.loads(cast(str, nodes.rows[later]["source_versions"])) == {
        "covers": {"version": 2, "first_seq": 5, "last_seq": 9},
    }
    assert json.loads(cast(str, nodes.rows[earlier]["source_versions"])) == {"covers": {}}
    assert all(nodes.rows[node]["parent_id"] == parent for node in (earlier, later))
    first_rows: Final = nodes.rows
    first_items: Final = items.rows
    first_updates: Final = items.updates

    repeated: Final = await backfill(cast("Prisma", client), dry_run=False)

    assert repeated == BackfillStats(sessions=2, items_text_filled=0, segment_nodes=2, session_nodes=1)
    assert items.rows == first_items
    assert items.updates == first_updates
    assert nodes.rows == first_rows
    assert tuple(where["node_id"] for where, _ in nodes.upserts) == (later, earlier, parent) * 2
