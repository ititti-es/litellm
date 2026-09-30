"""Backfill the hierarchical memory node table (litellm/sessions/MEMORY_DESIGN.md, step 2) and the
flat-text search column, over sessions already in the store.

    python -m litellm.proxy.memory_backfill [--database-url] [--dry-run]

Segment nodes are free: every pre-expiry or harness compaction summary already in the store
(`meta.kind == "compaction_summary"`, written by litellm/sessions/compaction.py:summary_meta, whose own
docstring calls it "the seed of a memory node") becomes one segment node, no new model calls. One session
node per session then lists its segment summaries (a flat concatenation, not a re-summarized model call --
cheap by design for this first pass). Project/scope levels, pgvector embeddings, and the live `expand()`
tool are not built here; see MEMORY_DESIGN.md.

Idempotent: item search_text is only written where NULL, and nodes are upserted by their deterministic id,
so re-running after new compactions land only adds what is new.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm.sessions.session import search_text

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models


_MAPPING: Final = TypeAdapter(Mapping[str, object])


class _SourceVersions(TypedDict):
    covers: ReadOnly[Mapping[str, object]]


class _ItemKey(TypedDict):
    session_id: ReadOnly[str]
    version: ReadOnly[int]
    seq: ReadOnly[int]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def segment_node_id(session_id: str, version: int, seq: int) -> str:
    return f"mem_seg_{_digest(f'{session_id}:{version}:{seq}')}"


def session_node_id(session_id: str) -> str:
    return f"mem_sess_{_digest(session_id)}"


@dataclass(frozen=True, slots=True)
class SegmentNode:
    node_id: str
    session_id: str
    summary: str
    policy_id: str | None
    model: str | None
    tokens: int | None
    span_version: int | None
    span_first_seq: int | None
    span_last_seq: int | None
    source_versions: str


def _as_mapping(value: object) -> Mapping[str, object]:
    return _MAPPING.validate_python(value if isinstance(value, Mapping) else MappingProxyType({}))


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def segment_from_item(
    session_id: str, version: int, seq: int, meta: Mapping[str, object], content: object
) -> SegmentNode | None:
    """Two shapes both tagged meta.kind == "compaction_summary": litellm's own live pre-expiry compaction
    (compaction.py:summary_meta) carries the summary text and the span it replaced directly in meta; a
    harness-imported one (opencode `summary` messages, openclaw compaction entries) is just a regular
    message item marked with this kind plus provenance -- its text is the item's own content, and the span
    it replaced is not recorded, so span_* stays unknown rather than guessed.
    """
    if meta.get("kind") != "compaction_summary":
        return None
    provenance: Final = _as_mapping(meta.get("provenance"))
    summary: Final = _as_str(meta.get("summary")) or (search_text(content) if isinstance(content, dict) else None)
    if not summary:
        return None
    covers: Final = _as_mapping(meta.get("covers"))
    usage: Final = _as_mapping(meta.get("usage"))
    source_versions: Final[_SourceVersions] = {"covers": covers}
    return SegmentNode(
        node_id=segment_node_id(session_id, version, seq),
        session_id=session_id,
        summary=summary,
        policy_id=_as_str(meta.get("policy")),
        model=_as_str(provenance.get("model")),
        tokens=_as_int(usage.get("output_tokens")),
        span_version=_as_int(covers.get("version")),
        span_first_seq=_as_int(covers.get("first_seq")),
        span_last_seq=_as_int(covers.get("last_seq")),
        source_versions=json.dumps(source_versions),
    )


def session_summary(segments: Sequence[SegmentNode]) -> str:
    """Cheap v1: list the segment summaries in span order, no re-summarizing model call"""
    ordered: Final = sorted(segments, key=lambda s: (s.span_version or 0, s.span_first_seq or 0))
    return "\n\n".join(f"- {s.summary}" for s in ordered)


@dataclass(frozen=True, slots=True)
class BackfillStats:
    sessions: int = 0
    items_text_filled: int = 0
    segment_nodes: int = 0
    session_nodes: int = 0


@asynccontextmanager
async def _client(url: str) -> AsyncGenerator[Prisma]:
    from prisma import Prisma
    from prisma.types import DatasourceOverride

    datasource: Final = DatasourceOverride(url=url)
    client: Final = Prisma(datasource=datasource)
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()


async def backfill(client: Prisma, *, dry_run: bool) -> BackfillStats:
    sessions: Final = await client.litellm_conversationsession.find_many()
    counts: Final = tuple([await _backfill_session(client, session, dry_run=dry_run) for session in sessions])
    return BackfillStats(
        sessions=len(sessions),
        items_text_filled=sum(count.items_text_filled for count in counts),
        segment_nodes=sum(count.segment_nodes for count in counts),
        session_nodes=sum(count.session_nodes for count in counts),
    )


async def _backfill_session(
    client: Prisma, session: prisma_models.LiteLLM_ConversationSession, *, dry_run: bool
) -> BackfillStats:
    items: Final = await client.litellm_conversationitem.find_many(
        where={"session_id": session.session_id, "version": session.version},  # mutable-ok: Prisma where
        order={"seq": "asc"},  # mutable-ok: Prisma order
    )
    text_filled: Final = sum(
        bool(search_text(item.content)) for item in items if item.search_text is None and isinstance(item.content, dict)
    )
    segments: Final = tuple(
        segment
        for item in items
        if (
            segment := segment_from_item(
                session.session_id, item.version, item.seq, _as_mapping(item.meta), item.content
            )
        )
        is not None
    )
    for item in items:
        if item.search_text is None:
            text = (search_text(item.content) if isinstance(item.content, dict) else None) or ""
            if not dry_run:
                item_key: _ItemKey = {
                    "session_id": item.session_id,
                    "version": item.version,
                    "seq": item.seq,
                }
                await client.litellm_conversationitem.update(
                    where={  # mutable-ok: Prisma where
                        "session_id_version_seq": item_key
                    },
                    data={"search_text": text},  # mutable-ok: Prisma data
                )

    for segment in segments:
        if not dry_run:
            await client.litellm_memorysummarynode.upsert(
                where={"node_id": segment.node_id},  # mutable-ok: Prisma where
                data={  # mutable-ok: Prisma upsert payload
                    "create": _segment_fields(segment),
                    "update": _segment_fields(segment),
                },
            )
    if segments:
        node_id: Final = session_node_id(session.session_id)
        if not dry_run:
            fields: Final = {  # mutable-ok: Prisma payload
                "session_id": session.session_id,
                "level": "session",
                "summary": session_summary(segments),
                "tokens": sum(s.tokens or 0 for s in segments) or None,
            }
            await client.litellm_memorysummarynode.upsert(
                where={"node_id": node_id},  # mutable-ok: Prisma where
                data={"create": {"node_id": node_id, **fields}, "update": fields},  # mutable-ok: Prisma payload
            )
            for segment in segments:
                await client.litellm_memorysummarynode.update(
                    where={"node_id": segment.node_id},  # mutable-ok: Prisma where
                    data={"parent_id": node_id},  # mutable-ok: Prisma data
                )
    return BackfillStats(
        sessions=1, items_text_filled=text_filled, segment_nodes=len(segments), session_nodes=int(bool(segments))
    )


def _segment_fields(segment: SegmentNode) -> Mapping[str, object]:
    return {  # mutable-ok: Prisma payload
        "node_id": segment.node_id,
        "session_id": segment.session_id,
        "level": "segment",
        "summary": segment.summary,
        "policy_id": segment.policy_id,
        "model": segment.model,
        "tokens": segment.tokens,
        "span_version": segment.span_version,
        "span_first_seq": segment.span_first_seq,
        "span_last_seq": segment.span_last_seq,
        "source_versions": segment.source_versions,
    }


async def _main(args: argparse.Namespace) -> int:
    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        sys.stderr.write("DATABASE_URL (or --database-url) is required\n")
        return 2
    async with _client(url) as client:
        stats: Final = await backfill(client, dry_run=args.dry_run)
    sys.stdout.write(
        f"sessions={stats.sessions} items_text_filled={stats.items_text_filled} "
        f"segment_nodes={stats.segment_nodes} session_nodes={stats.session_nodes}"
        f"{' (dry run, nothing written)' if args.dry_run else ''}\n"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.memory_backfill", description=(__doc__ or "").split("\n")[0]
    )
    parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    parser.add_argument("--dry-run", action="store_true", help="compute and count, write nothing")
    return parser


def main() -> int:
    return asyncio.run(_main(_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
