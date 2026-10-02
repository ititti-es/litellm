"""Collapse the copies that conversations under one client session id were stored as.

    python -m litellm.proxy.session_dedupe [--database-url URL] [--client-prefix ses_] [--apply --export DIR]

Before continue_client_session (litellm/sessions/manager.py) looked at every row recorded under a client
session id, a stream that did not own the id's derived row stored its whole history as a new row on every
turn: one opencode session sat in 784 rows, each a prefix of the next. A row is redundant when another row of
the same owner and client session id reproduces all of it, compared as `comparable` forms (the gateway's own
continuation test). The longest row of each conversation stays, so nothing is deleted that a kept row does not
say too; reasoning items, which `comparable` leaves out, are all a deleted copy can hold alone.

Dry run by default, in a read-only transaction: prints ids, counts and sizes, never message text. --apply first
writes every row it deletes, with its items and responses, to a gzipped JSON-lines file under --export, then
deletes one client session id's copies per transaction, skipping any row written to since the plan was made.
Only unpinned rows at version 0 with no memory nodes are considered.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, cast

from pydantic import JsonValue, TypeAdapter

from litellm.proxy.oauth_credential_store import libpq_url
from litellm.sessions.session import Item, comparable

_GROUPS_SQL: Final = """
SELECT s.owner, s.metadata->>'litellm_session_id', array_agg(s.session_id ORDER BY s.created_at)
FROM "LiteLLM_ConversationSession" s
WHERE s.source IS NULL
  AND s.version = 0
  AND NOT s.pinned
  AND s.metadata->>'litellm_session_id' LIKE %(prefix)s
  AND NOT EXISTS (SELECT 1 FROM "LiteLLM_MemorySummaryNode" m WHERE m.session_id = s.session_id)
GROUP BY 1, 2
HAVING count(*) > 1
ORDER BY 2
"""
_ITEMS_SQL: Final = """
SELECT content::text FROM "LiteLLM_ConversationItem" WHERE session_id = %(session_id)s AND version = 0 ORDER BY seq
"""
_SIZE_SQL: Final = """
SELECT count(*), coalesce(sum(pg_column_size(content)), 0) FROM "LiteLLM_ConversationItem"
WHERE session_id = ANY(%(ids)s)
"""
_EXPORT_SQL: Final = {
    "session": 'SELECT row_to_json(t)::text FROM "LiteLLM_ConversationSession" t WHERE session_id = %(session_id)s',
    "item": 'SELECT row_to_json(t)::text FROM "LiteLLM_ConversationItem" t WHERE session_id = %(session_id)s',
    "response": 'SELECT row_to_json(t)::text FROM "LiteLLM_ConversationResponse" t WHERE session_id = %(session_id)s',
}
_DELETE_SQL: Final = """
DELETE FROM "LiteLLM_ConversationSession"
WHERE session_id = ANY(%(ids)s) AND version = 0 AND NOT pinned AND updated_at <= %(planned_at)s
"""
_ITEM: Final = TypeAdapter(dict[str, JsonValue])


class _Cursor(Protocol):
    """The slice of a psycopg cursor used here, with rows typed as plain objects to be checked"""

    @property
    def rowcount(self) -> int: ...

    def execute(self, query: str, params: Mapping[str, object]) -> object: ...

    def fetchall(self) -> list[tuple[object, ...]]: ...  # mutable-ok: psycopg's row list

    def __enter__(self) -> _Cursor: ...

    def __exit__(self, *exc: object) -> None: ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor: ...

    def transaction(self) -> AbstractContextManager[object]: ...

    def close(self) -> None: ...


class _Options(argparse.Namespace):
    database_url: str | None = None
    client_prefix: str = "ses_"
    apply: bool = False
    export: str | None = None


@dataclass(frozen=True, slots=True)
class Row:
    session_id: str
    items: int
    forms: tuple[str, ...]
    """A digest per `comparable` form, in order"""


@dataclass(frozen=True, slots=True)
class Family:
    owner: str
    client_session_id: str
    rows: tuple[Row, ...]
    redundant: Mapping[str, str]
    """Each redundant row's id and the kept row that reproduces it"""


def redundant_rows(rows: Sequence[Row]) -> dict[str, str]:  # mutable-ok: built here, read-only after
    """Each row that another row reproduces in full, with that row; the longest rows are kept"""
    kept: list[Row] = []  # mutable-ok: grows as rows that nothing kept reproduces are found
    covered: dict[str, str] = {}  # mutable-ok: same
    for row in sorted(rows, key=lambda r: (-len(r.forms), -r.items, r.session_id)):
        keeper = next((k for k in kept if k.forms[: len(row.forms)] == row.forms), None)  # rebind-ok: per row
        if keeper is None:
            kept.append(row)
        else:
            covered[row.session_id] = keeper.session_id
    return covered


def form_digests(items: Sequence[Item]) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(json.dumps(form, ensure_ascii=False).encode()).hexdigest()[:20] for _, form in comparable(items)
    )


def _texts(rows: Iterable[tuple[object, ...]]) -> tuple[str, ...]:
    return tuple(value for row in rows if row and isinstance(value := row[0], str))


def _row(conn: _Connection, session_id: str) -> Row:
    with conn.cursor() as cursor:
        _ = cursor.execute(_ITEMS_SQL, {"session_id": session_id})
        items: Final = tuple(_ITEM.validate_json(content) for content in _texts(cursor.fetchall()))
    return Row(session_id, len(items), form_digests(items))


def _family(conn: _Connection, row: tuple[object, ...]) -> Family | None:
    owner, client_session_id, session_ids = (*row, None, None, None)[:3]
    if not isinstance(owner, str) or not isinstance(client_session_id, str) or not isinstance(session_ids, list):
        return None
    ids: Final = tuple(session_id for session_id in cast(list[object], session_ids) if isinstance(session_id, str))
    rows: Final = tuple(_row(conn, session_id) for session_id in ids)
    return Family(owner, client_session_id, rows, redundant_rows(rows))


def plan(conn: _Connection, prefix: str) -> tuple[Family, ...]:
    with conn.cursor() as cursor:
        _ = cursor.execute(_GROUPS_SQL, {"prefix": f"{prefix}%"})
        groups: Final = cursor.fetchall()
    return tuple(family for row in groups if (family := _family(conn, row)) is not None)


def _sizes(conn: _Connection, ids: Sequence[str]) -> tuple[int, int]:
    with conn.cursor() as cursor:
        _ = cursor.execute(_SIZE_SQL, {"ids": list(ids)})
        rows: Final = cursor.fetchall()
    count, size = rows[0][:2] if rows else (0, 0)
    return (count if isinstance(count, int) else 0), (size if isinstance(size, int) else 0)


def _report(conn: _Connection, families: Sequence[Family]) -> None:
    doomed: Final = [session_id for family in families for session_id in family.redundant]
    for family in families:
        kept = [row for row in family.rows if row.session_id not in family.redundant]  # rebind-ok: per family
        _ = sys.stdout.write(
            f"{family.client_session_id[:16]:16}  owner {family.owner[:8]}  rows {len(family.rows):4} -> {len(kept):2}"
            + f"  items {sum(r.items for r in family.rows):6} -> {sum(r.items for r in kept):5}\n"
        )
    items, size = _sizes(conn, doomed)
    rows: Final = sum(len(family.rows) for family in families)
    _ = sys.stdout.write(
        f"client session ids {len(families)}, rows {rows} -> {rows - len(doomed)}, "
        + f"items deleted {items}, content {size / 1e6:.0f} MB\n"
    )


def _apply(conn: _Connection, families: Sequence[Family], export: Path, planned_at: datetime) -> int:
    export.mkdir(parents=True, exist_ok=True)
    path: Final = export / f"session-dedupe-{planned_at:%Y%m%dT%H%M%SZ}.jsonl.gz"
    deleted = 0  # rebind-ok: running total
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for family in families:
            ids = list(family.redundant)  # rebind-ok: per family
            if not ids:
                continue
            with conn.transaction(), conn.cursor() as cursor:
                for session_id in ids:  # one row at a time: a family's copies can run to hundreds of MB
                    for kind, sql in _EXPORT_SQL.items():
                        _ = cursor.execute(sql, {"session_id": session_id})
                        for text in _texts(cursor.fetchall()):  # row_to_json text, written as it is
                            _ = out.write(f'{{"kind": {json.dumps(kind)}, "row": {text}}}\n')
                out.flush()
                _ = cursor.execute(_DELETE_SQL, {"ids": ids, "planned_at": planned_at.replace(tzinfo=None)})
                deleted += cursor.rowcount
    _ = sys.stdout.write(f"deleted {deleted} rows; their contents are in {path}\n")
    return deleted


def _connect(url: str, read_only: bool) -> _Connection:
    import psycopg

    options: Final = "-c default_transaction_read_only=on" if read_only else ""
    return cast(_Connection, psycopg.connect(libpq_url(url), options=options, autocommit=True))


def _main(args: _Options) -> int:
    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        _ = sys.stderr.write("DATABASE_URL (or --database-url) is required\n")
        return 2
    if args.apply and args.export is None:
        _ = sys.stderr.write("--apply needs --export DIR for the copies it deletes\n")
        return 2
    planned_at: Final = datetime.now(UTC)
    conn: Final = _connect(url, read_only=not args.apply)
    try:
        families: Final = plan(conn, args.client_prefix)
        _report(conn, families)
        if args.apply and args.export is not None:
            _ = _apply(conn, families, Path(args.export), planned_at)
        else:
            _ = sys.stdout.write("dry run: nothing deleted\n")
    finally:
        conn.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.session_dedupe", description=(__doc__ or "").split("\n")[0]
    )
    _ = parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    _ = parser.add_argument("--client-prefix", default="ses_", help="client session ids to consider (opencode's)")
    _ = parser.add_argument("--apply", action="store_true", help="delete the redundant rows (dry run otherwise)")
    _ = parser.add_argument("--export", help="directory for a gzipped JSON-lines copy of every deleted row")
    return parser


def main() -> int:
    return _main(_parser().parse_args(namespace=_Options()))


if __name__ == "__main__":
    sys.exit(main())
