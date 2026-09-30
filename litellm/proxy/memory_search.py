"""Hybrid-in-name-only (tsvector only; pgvector is deferred, see MEMORY_DESIGN.md) full-text search over
the session store: both raw item text (search_text, backfilled by memory_backfill) and memory summary
nodes (segment/session), merged and ranked by Postgres ts_rank.

    python -m litellm.proxy.memory_search "<query>" [--level segment|session] [--owner-key sk-...] [--limit N]

This is the CLI surface for "memory" -- consistent with the CLI-first direction that replaced MCP tool
access for local agents (tarot/soundcloud/house), not another MCP server.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import hash_token

if TYPE_CHECKING:
    from prisma import Prisma

_ITEMS_SQL: Final = """
SELECT i.session_id, i.seq, s.title, s.source,
       ts_rank(to_tsvector('english', COALESCE(i.search_text, '')), plainto_tsquery('english', $1)) AS rank,
       left(i.search_text, 200) AS snippet
FROM "LiteLLM_ConversationItem" i
JOIN "LiteLLM_ConversationSession" s ON s.session_id = i.session_id AND s.version = i.version
WHERE to_tsvector('english', COALESCE(i.search_text, '')) @@ plainto_tsquery('english', $1)
  AND ($2::text IS NULL OR s.owner = $2)
ORDER BY rank DESC
LIMIT $3
"""

_NODES_SQL: Final = """
SELECT n.node_id, n.session_id, n.level, s.title,
       ts_rank(to_tsvector('english', n.summary), plainto_tsquery('english', $1)) AS rank,
       left(n.summary, 200) AS snippet
FROM "LiteLLM_MemorySummaryNode" n
JOIN "LiteLLM_ConversationSession" s ON s.session_id = n.session_id
WHERE to_tsvector('english', n.summary) @@ plainto_tsquery('english', $1)
  AND ($2::text IS NULL OR n.level = $2)
  AND ($3::text IS NULL OR s.owner = $3)
ORDER BY rank DESC
LIMIT $4
"""


@dataclass(frozen=True, slots=True)
class Hit:
    kind: str  # "item" | "segment" | "session"
    session_id: str
    title: str | None
    rank: float
    snippet: str
    node_id: str | None = None
    seq: int | None = None


async def search(client: Prisma, query: str, *, level: str | None, owner: str | None, limit: int) -> tuple[Hit, ...]:
    item_rows: Final = await client.query_raw(_ITEMS_SQL, query, owner, limit) if level is None else ()
    node_rows: Final = await client.query_raw(_NODES_SQL, query, level, owner, limit)
    hits: Final = tuple(
        Hit(
            kind="item",
            session_id=r["session_id"],
            title=r["title"],
            rank=r["rank"],
            snippet=r["snippet"],
            seq=r["seq"],
        )
        for r in item_rows
    ) + tuple(
        Hit(
            kind=r["level"],
            session_id=r["session_id"],
            title=r["title"],
            rank=r["rank"],
            snippet=r["snippet"],
            node_id=r["node_id"],
        )
        for r in node_rows
    )
    return tuple(sorted(hits, key=lambda h: h.rank, reverse=True)[:limit])


def _format(hit: Hit) -> str:
    where: Final = f"seq={hit.seq}" if hit.node_id is None else hit.node_id
    title: Final = hit.title or "(untitled)"
    return f"[{hit.kind:<7}] {hit.rank:.4f}  {hit.session_id}  {title}  {where}\n    {hit.snippet}"


async def _main(args: argparse.Namespace) -> int:
    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        sys.stderr.write("DATABASE_URL (or --database-url) is required\n")
        return 2
    from prisma import Prisma
    from prisma.types import DatasourceOverride

    owner: Final = hash_token(args.owner_key) if args.owner_key else None
    datasource: Final = DatasourceOverride(url=url)
    client: Final = Prisma(datasource=datasource)
    await client.connect()
    try:
        hits: Final = await search(client, args.query, level=args.level, owner=owner, limit=args.limit)
    finally:
        await client.disconnect()
    if not hits:
        sys.stdout.write("no matches\n")
        return 1
    for hit in hits:
        sys.stdout.write(_format(hit) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.memory_search", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("query")
    parser.add_argument(
        "--level",
        choices=("segment", "session"),
        help="restrict to one memory-node level (omit to also search raw item text)",
    )
    parser.add_argument("--owner-key", help="restrict to sessions owned by this virtual key")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    return parser


def main() -> int:
    return asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
