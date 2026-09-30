from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, cast

import pytest

from litellm.proxy.memory_search import search

if TYPE_CHECKING:
    from prisma import Prisma


class _SearchClient:
    def __init__(self) -> None:
        self.calls: tuple[tuple[object, ...], ...] = ()

    async def query_raw(self, _sql: str, *parameters: object) -> tuple[Mapping[str, object], ...]:
        self.calls += (parameters,)
        if len(parameters) == 3:
            query, owner, limit = parameters
            return (
                (
                    {
                        "session_id": "synthetic",
                        "seq": 1,
                        "title": None,
                        "source": "live",
                        "rank": 0.6,
                        "snippet": "item",
                    },
                )
                if query == "marker" and owner == "owner" and limit == 2
                else ()
            )
        query, level, owner, limit = parameters
        rows: Final = (
            {
                "session_id": "synthetic",
                "node_id": "segment",
                "level": "segment",
                "title": None,
                "rank": 0.9,
                "snippet": "segment",
            },
            {
                "session_id": "synthetic",
                "node_id": "session",
                "level": "session",
                "title": None,
                "rank": 0.2,
                "snippet": "session",
            },
        )
        return (
            tuple(row for row in rows if level is None or row["level"] == level)
            if query == "marker" and owner == "owner" and limit == 2
            else ()
        )


@pytest.mark.asyncio
async def test_search_merges_and_ranks_items_and_nodes_before_applying_limit() -> None:
    client: Final = _SearchClient()
    hits: Final = await search(cast("Prisma", client), "marker", level=None, owner="owner", limit=2)

    assert tuple(hit.kind for hit in hits) == ("segment", "item")
    assert tuple(hit.snippet for hit in hits) == ("segment", "item")
    assert hits[0].node_id == "segment" and hits[1].seq == 1
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_node_level_search_excludes_raw_items() -> None:
    client: Final = _SearchClient()
    hits: Final = await search(cast("Prisma", client), "marker", level="session", owner="owner", limit=2)

    assert tuple(hit.kind for hit in hits) == ("session",)
    assert hits[0].node_id == "session" and hits[0].seq is None
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_search_preserves_empty_results_for_another_owner_or_query() -> None:
    client: Final = _SearchClient()
    assert await search(cast("Prisma", client), "marker", level=None, owner="another-owner", limit=2) == ()
    assert await search(cast("Prisma", client), "absent", level=None, owner="owner", limit=2) == ()
