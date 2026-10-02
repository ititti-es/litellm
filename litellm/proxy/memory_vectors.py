"""Semantic search with rerank over the session store: the pgvector half of the hybrid retrieval that
litellm/sessions/MEMORY_DESIGN.md plans, next to the tsvector-only memory_search.

    python -m litellm.proxy.memory_vectors init-schema
    python -m litellm.proxy.memory_vectors index [--dry-run] [--max-chars N]
    python -m litellm.proxy.memory_vectors search "<query>" [--user ID] [--team ID] [--agent NAME] [--harness NAME]
        [--owner-key sk-...] [--limit N]

Every item's searchable text (litellm/sessions/session.py:search_text) is labeled with who said it, split into
overlapping chunks and embedded once per distinct text: a session's versions and opencode retries copy the same
history many times over, so chunks are keyed by the text's digest and items only link to that digest. Search
fuses a vector leg and a full-text leg over the chunks by reciprocal rank, then reranks the fused candidates.

The tables are not in schema.prisma: Prisma cannot express halfvec or HNSW, and keeping them out of the
migrations means a proxy restart never depends on the vector extension, which only a superuser can create.
`init-schema` is idempotent and only adds tables of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby, islice
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from psycopg.rows import class_row
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from typing_extensions import ReadOnly

from litellm.proxy._types import hash_token
from litellm.sessions.session import Item, search_text

if TYPE_CHECKING:
    import psycopg

EMBEDDING_MODEL: Final = "Qwen/Qwen3-Embedding-8B"
RERANK_MODEL: Final = "Qwen/Qwen3-Reranker-8B"
DIMENSIONS: Final = 1024
"""Qwen3-Embedding is Matryoshka-trained, so a 1024 prefix of its 4096 dims keeps most of the quality, and HNSW
indexes halfvec only up to 4000 dims. SCHEMA_SQL spells the same number out, since its SQL must stay a literal"""
QUERY_INSTRUCTION: Final = "Given a search query, retrieve passages from past conversations that answer it"
DEEPINFRA_BASE_URL: Final = "https://api.deepinfra.com/v1"

CHUNK_CHARS: Final = 2000
CHUNK_OVERLAP: Final = 200
MAX_ITEM_CHARS: Final = 32_000
"""Only the head of a huge tool output is indexed: a 250k-character log is mostly noise to a search"""
EMBED_BATCH_CHARS: Final = 120_000
EMBED_BATCH_SIZE: Final = 64
ITEM_SCAN_BATCH: Final = 2000
RRF_K: Final = 60
RETRY_ATTEMPTS: Final = 8
RETRYABLE_STATUSES: Final = frozenset((429, 500, 502, 503, 504))
LEG_LIMIT: Final = 100
RERANK_CANDIDATES: Final = 40
RERANK_DOC_CHARS: Final = 4000

SCHEMA_SQL: Final = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS "LiteLLM_ConversationChunk" (
    "text_sha"   BYTEA NOT NULL,
    "chunk_no"   INTEGER NOT NULL,
    "model"      TEXT NOT NULL,
    "content"    TEXT NOT NULL,
    "embedding"  halfvec(1024) NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "LiteLLM_ConversationChunk_pkey" PRIMARY KEY ("text_sha", "chunk_no")
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ConversationChunk_embedding_hnsw_idx" ON "LiteLLM_ConversationChunk"
    USING hnsw ("embedding" halfvec_cosine_ops);

CREATE INDEX IF NOT EXISTS "LiteLLM_ConversationChunk_content_gin_idx" ON "LiteLLM_ConversationChunk"
    USING GIN (to_tsvector('english', "content"));

CREATE TABLE IF NOT EXISTS "LiteLLM_ConversationItemText" (
    "session_id" TEXT NOT NULL,
    "version"    INTEGER NOT NULL,
    "seq"        INTEGER NOT NULL,
    "text_sha"   BYTEA,
    CONSTRAINT "LiteLLM_ConversationItemText_pkey" PRIMARY KEY ("session_id", "version", "seq"),
    CONSTRAINT "LiteLLM_ConversationItemText_item_fkey" FOREIGN KEY ("session_id", "version", "seq")
        REFERENCES "LiteLLM_ConversationItem" ("session_id", "version", "seq") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS "LiteLLM_ConversationItemText_text_sha_idx" ON "LiteLLM_ConversationItemText" ("text_sha");
"""

_UNLINKED_ITEMS_SQL: Final = """
SELECT i.session_id, i.version, i.seq,
       CASE WHEN jsonb_typeof(i.content) = 'object' THEN i.content END AS content
FROM "LiteLLM_ConversationItem" i
WHERE NOT EXISTS (
    SELECT 1 FROM "LiteLLM_ConversationItemText" t
    WHERE t.session_id = i.session_id AND t.version = i.version AND t.seq = i.seq
)
LIMIT %s
"""

_KNOWN_SHAS_SQL: Final = """
SELECT DISTINCT text_sha FROM "LiteLLM_ConversationChunk" WHERE text_sha = ANY(%s) AND model = %s
"""

_INSERT_CHUNK_SQL: Final = """
INSERT INTO "LiteLLM_ConversationChunk" (text_sha, chunk_no, model, content, embedding)
VALUES (%s, %s, %s, %s, %s::halfvec)
ON CONFLICT (text_sha, chunk_no) DO NOTHING
"""

_INSERT_LINK_SQL: Final = """
INSERT INTO "LiteLLM_ConversationItemText" (session_id, version, seq, text_sha)
VALUES (%s, %s, %s, %s)
ON CONFLICT (session_id, version, seq) DO NOTHING
"""

MASTER_KEY_OWNER: Final = "litellm_proxy_master_key"
"""The owner the store records for turns made with the proxy master key, which has no key row or user of its own"""

_SESSION_IN_SCOPE: Final = """
    (%(owner)s::text IS NULL OR s.owner = %(owner)s)
    AND (%(user)s::text IS NULL OR COALESCE(
        v.user_id, CASE WHEN s.owner = 'litellm_proxy_master_key' THEN %(master_key_user)s::text END
    ) = %(user)s)
    AND (%(team)s::text IS NULL OR v.team_id = %(team)s)
    AND (%(agent)s::text IS NULL OR COALESCE(s.metadata->>'agent', v.metadata->>'agent') = %(agent)s)
    AND (%(harness)s::text IS NULL OR COALESCE(s.metadata->>'harness', s.source, v.metadata->>'harness') = %(harness)s)
"""

_OWNER_FILTER: Final = f"""
(%(unscoped)s OR EXISTS (
    SELECT 1 FROM "LiteLLM_ConversationItemText" t
    JOIN "LiteLLM_ConversationSession" s ON s.session_id = t.session_id
    LEFT JOIN "LiteLLM_VerificationToken" v ON v.token = s.owner
    WHERE t.text_sha = c.text_sha AND {_SESSION_IN_SCOPE}
))
"""

_VECTOR_LEG_SQL: Final = f"""
SELECT c.text_sha, c.chunk_no, c.content
FROM "LiteLLM_ConversationChunk" c
WHERE c.model = %(model)s AND {_OWNER_FILTER}
ORDER BY c.embedding <=> %(vector)s::halfvec
LIMIT %(limit)s
"""

_TEXT_LEG_SQL: Final = f"""
SELECT c.text_sha, c.chunk_no, c.content
FROM "LiteLLM_ConversationChunk" c
WHERE to_tsvector('english', c.content) @@ websearch_to_tsquery('english', %(query)s)
  AND c.model = %(model)s AND {_OWNER_FILTER}
ORDER BY ts_rank(to_tsvector('english', c.content), websearch_to_tsquery('english', %(query)s)) DESC
LIMIT %(limit)s
"""

_OCCURRENCE_SQL: Final = f"""
SELECT DISTINCT ON (t.text_sha) t.text_sha, t.session_id, t.seq, s.title, s.source,
       COALESCE(s.metadata->>'agent', v.metadata->>'agent') AS agent,
       COALESCE(s.metadata->>'harness', s.source, v.metadata->>'harness') AS harness
FROM "LiteLLM_ConversationItemText" t
JOIN "LiteLLM_ConversationSession" s ON s.session_id = t.session_id
LEFT JOIN "LiteLLM_VerificationToken" v ON v.token = s.owner
WHERE t.text_sha = ANY(%(shas)s) AND {_SESSION_IN_SCOPE}
ORDER BY t.text_sha, (t.version = s.version) DESC, s.updated_at DESC, t.seq
"""


@dataclass(frozen=True, slots=True)
class Scope:
    """Which sessions a search may return; every field left None is unrestricted"""

    owner: str | None = None
    user: str | None = None
    team: str | None = None
    agent: str | None = None
    harness: str | None = None
    master_key_user: str | None = None
    """The human the master key's sessions count as, since that key has no user of its own"""

    def params(self) -> Mapping[str, object]:
        restricting: Final = (self.owner, self.user, self.team, self.agent, self.harness)
        return MappingProxyType(
            {
                "owner": self.owner,
                "user": self.user,
                "team": self.team,
                "agent": self.agent,
                "harness": self.harness,
                "master_key_user": self.master_key_user,
                "unscoped": all(value is None for value in restricting),
            }
        )


EVERYTHING: Final = Scope()


def labeled_text(item: Item) -> str | None:
    """The item's searchable text, prefixed with who produced it so a chunk reads on its own"""
    text: Final = search_text(item)
    if not text or not text.strip():
        return None
    kind: Final = item.get("type", "message")
    label: Final = (
        f"tool call {item.get('name') or ''}".rstrip()
        if kind == "function_call"
        else "tool output"
        if kind == "function_call_output"
        else str(item.get("role") or "message")
    )
    return f"{label}: {text.replace(chr(0), '')}"


def text_digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()


def _chunk_end(text: str, start: int, size: int) -> int:
    hard_end: Final = min(start + size, len(text))
    if hard_end == len(text):
        return hard_end
    window: Final = text[start + size // 2 : hard_end]
    breaks: Final = tuple(window.rfind(separator) for separator in ("\n\n", "\n", ". ", " "))
    best: Final = next((position for position in breaks if position >= 0), -1)
    return hard_end if best < 0 else start + size // 2 + best + 1


def _chunk_starts(text: str, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    start = 0  # rebind-ok: walks the text one window at a time
    while start < len(text):
        end = _chunk_end(text, start, size)
        yield start, end
        if end >= len(text):
            return
        start = max(end - overlap, start + 1)


def chunk(
    text: str, *, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP, cap: int = MAX_ITEM_CHARS
) -> tuple[str, ...]:
    """Overlapping windows of at most `size` characters over the first `cap` characters, cut at a paragraph,
    line, sentence or word boundary when one falls in the window's second half"""
    head: Final = text[:cap]
    return tuple(piece for start, end in _chunk_starts(head, size, overlap) if (piece := head[start:end].strip()))


def reciprocal_rank_fusion(
    legs: Sequence[Sequence[tuple[bytes, int]]], k: int = RRF_K
) -> tuple[tuple[bytes, int], ...]:
    scores: Final = MappingProxyType(
        {
            key: sum(1 / (k + rank) for leg in legs for rank, other in enumerate(leg, start=1) if other == key)
            for key in dict.fromkeys(key for leg in legs for key in leg)
        }
    )
    return tuple(sorted(scores, key=lambda key: scores[key], reverse=True))


def embedding_batches(texts: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Consecutive slices of `texts` bounded by both count and total characters, never empty"""
    batch: tuple[str, ...] = ()  # rebind-ok: the open batch grows until a limit closes it
    for text in texts:
        if batch and (len(batch) >= EMBED_BATCH_SIZE or sum(map(len, batch)) + len(text) > EMBED_BATCH_CHARS):
            yield batch
            batch = ()  # rebind-ok: starts the next batch
        batch = (*batch, text)  # rebind-ok: grows the open batch
    if batch:
        yield batch


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class Reranker(Protocol):
    def __call__(self, query: str, documents: Sequence[str]) -> tuple[float, ...]: ...


class _EmbeddingBody(TypedDict):
    model: ReadOnly[str]
    input: ReadOnly[tuple[str, ...]]
    dimensions: ReadOnly[int]
    encoding_format: ReadOnly[str]


class _RerankBody(TypedDict):
    queries: ReadOnly[tuple[str]]
    documents: ReadOnly[tuple[str, ...]]
    instruction: ReadOnly[str]


class _EmbeddingRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    index: int
    embedding: tuple[float, ...]


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_EmbeddingRow, ...]


class _RerankResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    scores: tuple[float, ...]


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after: Final = str(response.headers.get("retry-after") or "")
    return float(retry_after) if retry_after.isdigit() else min(2.0 * 2.0**attempt, 60.0)


@dataclass(frozen=True, slots=True)
class DeepInfra:
    client: httpx.Client
    api_key: str
    sleep: Callable[[float], None] = time.sleep

    def _post(self, path: str, body: _EmbeddingBody | _RerankBody) -> bytes:
        response: Final = self._send(path, body, attempt=0)
        _ = response.raise_for_status()
        return response.content

    def _send(self, path: str, body: _EmbeddingBody | _RerankBody, *, attempt: int) -> httpx.Response:
        response: Final = self.client.post(
            f"{DEEPINFRA_BASE_URL}{path}",
            json=body,
            headers=MappingProxyType({"Authorization": f"Bearer {self.api_key}"}),
        )
        if response.status_code not in RETRYABLE_STATUSES or attempt == RETRY_ATTEMPTS - 1:
            return response
        self.sleep(_retry_delay(response, attempt))
        return self._send(path, body, attempt=attempt + 1)

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        body: Final[_EmbeddingBody] = {
            "model": EMBEDDING_MODEL,
            "input": tuple(texts),
            "dimensions": DIMENSIONS,
            "encoding_format": "float",
        }
        rows: Final = _EmbeddingResponse.model_validate_json(self._post("/openai/embeddings", body)).data
        if len(rows) != len(texts):
            raise ValueError(f"{EMBEDDING_MODEL} returned {len(rows)} vectors for {len(texts)} inputs")
        return tuple(row.embedding for row in sorted(rows, key=lambda row: row.index))

    def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        body: Final[_RerankBody] = {
            "queries": (query,),
            "documents": tuple(documents),
            "instruction": QUERY_INSTRUCTION,
        }
        scores: Final = _RerankResponse.model_validate_json(self._post(f"/inference/{RERANK_MODEL}", body)).scores
        if len(scores) != len(documents):
            raise ValueError(f"{RERANK_MODEL} returned {len(scores)} scores for {len(documents)} documents")
        return scores


def psycopg_url(url: str) -> str:
    """DATABASE_URL with only the query parameters libpq knows: the proxy appends Prisma pool settings
    (connection_limit, pool_timeout, max_idle_connection_lifetime, ...) that libpq rejects"""
    from psycopg import pq

    known: Final = frozenset(option.keyword.decode() for option in pq.Conninfo.get_defaults())
    parts: Final = urlsplit(url)
    kept: Final = tuple((key, value) for key, value in parse_qsl(parts.query) if key in known)
    return urlunsplit(parts._replace(query=urlencode(kept)))


def query_text(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{value:.6g}" for value in vector) + "]"


@dataclass(frozen=True, slots=True)
class IndexStats:
    items: int
    texts: int
    chunks: int
    chars: int


ITEM_CONTENT: Final = TypeAdapter[dict[str, JsonValue] | None](dict[str, JsonValue] | None)


@dataclass(frozen=True, slots=True)
class _UnlinkedItem:
    session_id: str
    version: int
    seq: int
    content: object


@dataclass(frozen=True, slots=True)
class _LabeledItem:
    session_id: str
    version: int
    seq: int
    text: str | None


@dataclass(frozen=True, slots=True)
class _KnownText:
    text_sha: bytes


@dataclass(frozen=True, slots=True)
class _ChunkRow:
    text_sha: bytes
    chunk_no: int
    content: str


@dataclass(frozen=True, slots=True)
class _Occurrence:
    text_sha: bytes
    session_id: str
    seq: int
    title: str | None
    source: str | None
    agent: str | None
    harness: str | None


def _labeled(row: _UnlinkedItem) -> _LabeledItem:
    parsed: Final = ITEM_CONTENT.validate_python(row.content)
    content: Final = MappingProxyType(parsed) if parsed is not None else None
    return _LabeledItem(
        session_id=row.session_id,
        version=row.version,
        seq=row.seq,
        text=labeled_text(content) if content is not None else None,
    )


def _missing_chunks(conn: psycopg.Connection, texts: Mapping[bytes, str]) -> Mapping[bytes, tuple[str, ...]]:
    with conn.cursor(row_factory=class_row(_KnownText)) as cursor:
        known: Final = frozenset(
            row.text_sha
            for row in cursor.execute(
                _KNOWN_SHAS_SQL,
                (list(texts), EMBEDDING_MODEL),  # mutable-ok: psycopg adapts only a list to a Postgres array
            ).fetchall()
        )
    return MappingProxyType({sha: chunk(text) for sha, text in texts.items() if sha not in known})


def _store_chunks(conn: psycopg.Connection, embed: Embedder, chunks: Mapping[bytes, tuple[str, ...]]) -> None:
    keyed: Final = tuple((sha, number, piece) for sha, pieces in chunks.items() for number, piece in enumerate(pieces))
    vectors: Final = tuple(
        vector for batch in embedding_batches(tuple(piece for _, _, piece in keyed)) for vector in embed(batch)
    )
    with conn.cursor() as cursor:
        cursor.executemany(
            _INSERT_CHUNK_SQL,
            tuple(
                (sha, number, EMBEDDING_MODEL, piece, vector_literal(vector))
                for (sha, number, piece), vector in zip(keyed, vectors, strict=True)
            ),
        )


def index_pass(conn: psycopg.Connection, embed: Embedder, *, dry_run: bool, max_chars: int | None) -> IndexStats:
    """Links and embeds one batch of not-yet-linked items. Chunks are stored before the links that point at them,
    so an interrupted pass leaves those items unlinked and the next pass picks them up again."""
    with conn.cursor(row_factory=class_row(_UnlinkedItem)) as cursor:
        rows: Final = cursor.execute(_UNLINKED_ITEMS_SQL, (ITEM_SCAN_BATCH,)).fetchall()
    labeled: Final = tuple(_labeled(row) for row in rows)
    texts: Final = MappingProxyType({text_digest(item.text): item.text for item in labeled if item.text})
    missing: Final = _missing_chunks(conn, texts)
    chars: Final = sum(len(piece) for pieces in missing.values() for piece in pieces)
    stats: Final = IndexStats(items=len(rows), texts=len(missing), chunks=sum(map(len, missing.values())), chars=chars)
    if dry_run or (max_chars is not None and chars > max_chars):
        return stats
    with conn.transaction():
        _store_chunks(conn, embed, missing)
        with conn.cursor() as cursor:
            cursor.executemany(
                _INSERT_LINK_SQL,
                tuple(
                    (item.session_id, item.version, item.seq, text_digest(item.text) if item.text else None)
                    for item in labeled
                ),
            )
    return stats


@dataclass(frozen=True, slots=True)
class Hit:
    score: float
    session_id: str
    seq: int
    title: str | None
    source: str | None
    agent: str | None
    harness: str | None
    chunk_no: int
    content: str


def _legs(
    conn: psycopg.Connection, query: str, vector: Sequence[float], scope: Scope
) -> tuple[tuple[_ChunkRow, ...], tuple[_ChunkRow, ...]]:
    params: Final = MappingProxyType({**scope.params(), "model": EMBEDDING_MODEL, "limit": LEG_LIMIT, "query": query})
    with conn.transaction(), conn.cursor(row_factory=class_row(_ChunkRow)) as cursor:
        _ = cursor.execute("SET LOCAL hnsw.ef_search = 200")
        _ = cursor.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        by_vector: Final = tuple(
            cursor.execute(_VECTOR_LEG_SQL, MappingProxyType({**params, "vector": vector_literal(vector)}))
        )
        by_text: Final = tuple(cursor.execute(_TEXT_LEG_SQL, params))
    return by_vector, by_text


def _occurrences(conn: psycopg.Connection, shas: Sequence[bytes], scope: Scope) -> Mapping[bytes, _Occurrence]:
    with conn.cursor(row_factory=class_row(_Occurrence)) as cursor:
        rows: Final = cursor.execute(
            _OCCURRENCE_SQL,
            {**scope.params(), "shas": list(shas)},  # mutable-ok: psycopg adapts only a list to a Postgres array
        ).fetchall()
    return MappingProxyType({row.text_sha: row for row in rows})


def search(
    conn: psycopg.Connection, embed: Embedder, rerank: Reranker, query: str, *, scope: Scope, limit: int
) -> tuple[Hit, ...]:
    legs: Final = _legs(conn, query, embed((query_text(query),))[0], scope)
    contents: Final = MappingProxyType({(row.text_sha, row.chunk_no): row.content for leg in legs for row in leg})
    fused: Final = reciprocal_rank_fusion(tuple(tuple((row.text_sha, row.chunk_no) for row in leg) for leg in legs))
    candidates: Final = fused[:RERANK_CANDIDATES]
    if not candidates:
        return ()
    scores: Final = rerank(query, tuple(contents[key][:RERANK_DOC_CHARS] for key in candidates))
    by_text: Final = sorted(zip(scores, candidates, strict=True), key=lambda pair: (pair[1][0], -pair[0]))
    best_per_text: Final = tuple(next(group) for _, group in groupby(by_text, key=lambda pair: pair[1][0]))
    top: Final = tuple(islice(sorted(best_per_text, key=lambda pair: pair[0], reverse=True), limit))
    occurrences: Final = _occurrences(conn, tuple(sha for _, (sha, _) in top), scope)
    return tuple(
        Hit(
            score=score,
            session_id=occurrences[sha].session_id,
            seq=occurrences[sha].seq,
            title=occurrences[sha].title,
            source=occurrences[sha].source,
            agent=occurrences[sha].agent,
            harness=occurrences[sha].harness,
            chunk_no=number,
            content=contents[(sha, number)],
        )
        for score, (sha, number) in top
        if sha in occurrences
    )


def _format(hit: Hit) -> str:
    snippet: Final = " ".join(hit.content.split())[:240]
    title: Final = hit.title or "(untitled)"
    return f"{hit.score:.3f}  {hit.session_id}  {title}  seq={hit.seq} [{hit.source or '-'}]\n    {snippet}"


class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    command: Literal["init-schema", "index", "search"]
    database_url: str | None = None
    dry_run: bool = False
    max_chars: int | None = None
    query: str = ""
    owner_key: str | None = None
    user: str | None = None
    team: str | None = None
    agent: str | None = None
    harness: str | None = None
    master_key_user: str | None = None
    limit: int = 10


def _connect(args: _Args) -> psycopg.Connection:
    import psycopg

    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL (or --database-url) is required")
    return psycopg.connect(psycopg_url(url), autocommit=True)


def _deepinfra() -> DeepInfra:
    key: Final = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise SystemExit("DEEPINFRA_API_KEY is required")
    return DeepInfra(client=httpx.Client(timeout=120), api_key=key)


def _added(left: IndexStats, right: IndexStats) -> IndexStats:
    return IndexStats(
        items=left.items + right.items,
        texts=left.texts + right.texts,
        chunks=left.chunks + right.chunks,
        chars=left.chars + right.chars,
    )


def _describe(stats: IndexStats) -> str:
    return f"{stats.items} items, {stats.texts} new texts, {stats.chunks} chunks, {stats.chars} chars"


def _run_index(conn: psycopg.Connection, args: _Args) -> int:
    deepinfra: Final = _deepinfra()
    if args.dry_run:
        sys.stdout.write(f"next batch: {_describe(index_pass(conn, deepinfra.embed, dry_run=True, max_chars=None))}\n")
        return 0
    totals = IndexStats(items=0, texts=0, chunks=0, chars=0)  # rebind-ok: running totals across passes
    while (stats := index_pass(conn, deepinfra.embed, dry_run=False, max_chars=args.max_chars)).items:
        if args.max_chars is not None and stats.chars > args.max_chars:
            sys.stderr.write(f"stopping: next batch needs {stats.chars} chars, over --max-chars {args.max_chars}\n")
            return 3
        totals = _added(totals, stats)  # rebind-ok: running totals across passes
        sys.stdout.write(f"indexed {_describe(totals)}\n")
        sys.stdout.flush()
    return 0


def _run_search(conn: psycopg.Connection, args: _Args) -> int:
    deepinfra: Final = _deepinfra()
    scope: Final = Scope(
        owner=hash_token(args.owner_key) if args.owner_key else None,
        user=args.user,
        team=args.team,
        agent=args.agent,
        harness=args.harness,
        master_key_user=args.master_key_user or os.environ.get("MEMORY_MASTER_KEY_USER"),
    )
    hits: Final = search(conn, deepinfra.embed, deepinfra.rerank, args.query, scope=scope, limit=args.limit)
    if not hits:
        sys.stdout.write("no matches\n")
        return 1
    sys.stdout.write("".join(_format(hit) + "\n" for hit in hits))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.memory_vectors", description=(__doc__ or "").split("\n")[0]
    )
    _ = parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    _ = commands.add_parser("init-schema", help="create the vector extension, chunk and link tables if missing")
    index: Final = commands.add_parser("index", help="embed every item not yet linked to its chunks")
    _ = index.add_argument("--dry-run", action="store_true", help="report the next batch without embedding")
    _ = index.add_argument("--max-chars", type=int, help="stop before a batch that would embed more than this")
    search_command: Final = commands.add_parser("search", help="hybrid vector and full-text search, reranked")
    _ = search_command.add_argument("query")
    _ = search_command.add_argument("--owner-key", help="restrict to sessions owned by this virtual key")
    _ = search_command.add_argument("--user", help="restrict to sessions whose key belongs to this user id")
    _ = search_command.add_argument("--team", help="restrict to sessions whose key belongs to this team id")
    _ = search_command.add_argument("--agent", help="restrict to sessions recorded with this agent")
    _ = search_command.add_argument("--harness", help="restrict to sessions recorded with this harness")
    _ = search_command.add_argument(
        "--master-key-user", help="user the master key's sessions count as (default $MEMORY_MASTER_KEY_USER)"
    )
    _ = search_command.add_argument("--limit", type=int, default=10)
    return parser


def main() -> int:
    args: Final = _Args.model_validate(vars(_parser().parse_args()))
    with _connect(args) as conn:
        if args.command == "init-schema":
            _ = conn.execute(SCHEMA_SQL)
            return 0
        if args.command == "index":
            return _run_index(conn, args)
        return _run_search(conn, args)


if __name__ == "__main__":
    sys.exit(main())
