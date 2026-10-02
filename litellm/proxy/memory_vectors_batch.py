"""Bulk backfill for memory_vectors through DeepInfra's OpenAI-style Batch API, at batch pricing instead of one
synchronous request at a time.

    python -m litellm.proxy.memory_vectors_batch export DIR
    python -m litellm.proxy.memory_vectors_batch submit DIR
    python -m litellm.proxy.memory_vectors_batch status DIR
    python -m litellm.proxy.memory_vectors_batch import DIR

`export` writes one embeddings request per chunk group for every chunk not yet stored, plus a manifest naming each
input by the digest of its text. Nothing in DIR holds transcript text except the request files DeepInfra receives.
`import` loads the returned vectors keyed by those digests and then runs the ordinary memory_vectors index passes,
whose embedder answers from the loaded vectors and falls back to a live call for anything the batch did not return,
so links and chunks are written exactly as a synchronous backfill would write them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TextIO

import httpx
from psycopg.rows import class_row
from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.proxy.memory_vectors import (
    DEEPINFRA_BASE_URL,
    DIMENSIONS,
    EMBEDDING_MODEL,
    ITEM_CONTENT,
    Embedder,
    IndexStats,
    chunk,
    embedding_batches,
    index_pass,
    labeled_text,
    psycopg_url,
    text_digest,
)

if TYPE_CHECKING:
    import psycopg

REQUESTS_PER_FILE: Final = 500
"""At most 60 MB of request text per input file (500 requests of up to 120k characters), held in memory while written"""
_ENDPOINT: Final = "/v1/embeddings"
_BATCH_IDS: Final = TypeAdapter(tuple[str, ...])
_HALF_TUPLE: Final = TypeAdapter(tuple[float, ...])
_HALF: Final = struct.Struct(f"<{DIMENSIONS}e")
"""Vectors are held as packed half floats, 2 KB each, since the whole backfill's results sit in memory at once"""

_ALL_UNLINKED_SQL: Final = """
SELECT i.session_id, i.version, i.seq,
       CASE WHEN jsonb_typeof(i.content) = 'object' THEN i.content END AS content
FROM "LiteLLM_ConversationItem" i
WHERE NOT EXISTS (
    SELECT 1 FROM "LiteLLM_ConversationItemText" t
    WHERE t.session_id = i.session_id AND t.version = i.version AND t.seq = i.seq
)
"""

_ALL_KNOWN_SHAS_SQL: Final = """SELECT DISTINCT text_sha FROM "LiteLLM_ConversationChunk" WHERE model = %s"""


def chunk_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class _Row:
    session_id: str
    version: int
    seq: int
    content: object


@dataclass(frozen=True, slots=True)
class _Known:
    text_sha: bytes


def _unlinked_texts(conn: psycopg.Connection) -> Iterator[str]:
    with conn.transaction(), conn.cursor(name="unlinked_items", row_factory=class_row(_Row)) as cursor:
        cursor.itersize = 500
        for row in cursor.execute(_ALL_UNLINKED_SQL):
            parsed = ITEM_CONTENT.validate_python(row.content)  # rebind-ok: one item per iteration
            text = labeled_text(MappingProxyType(parsed)) if parsed is not None else None  # rebind-ok: per item
            if text:
                yield text


def missing_chunks(conn: psycopg.Connection) -> Iterator[str]:
    """Every distinct chunk of every distinct unlinked text whose chunks are not stored yet, streamed"""
    with conn.cursor(row_factory=class_row(_Known)) as cursor:
        known: Final = frozenset(row.text_sha for row in cursor.execute(_ALL_KNOWN_SHAS_SQL, (EMBEDDING_MODEL,)))
    seen_texts: Final[set[bytes]] = set()  # mutable-ok: dedupe set over a stream too large to hold as text
    seen_chunks: Final[set[str]] = set()  # mutable-ok: dedupe set over a stream too large to hold as text
    for text in _unlinked_texts(conn):
        sha = text_digest(text)  # rebind-ok: one text per iteration
        if sha in known or sha in seen_texts:
            continue
        seen_texts.add(sha)
        for piece in chunk(text):
            digest = chunk_digest(piece)  # rebind-ok: one chunk per iteration
            if digest not in seen_chunks:
                seen_chunks.add(digest)
                yield piece


@dataclass(frozen=True, slots=True)
class ExportStats:
    files: int
    requests: int
    chunks: int
    chars: int


def _manifest_line(custom_id: str, inputs: Sequence[str]) -> str:
    return json.dumps({"custom_id": custom_id, "digests": tuple(map(chunk_digest, inputs))}) + "\n"  # mutable-ok: JSON


def _request_line(custom_id: str, inputs: Sequence[str]) -> str:
    body: Final = {  # mutable-ok: JSON request body
        "model": EMBEDDING_MODEL,
        "input": tuple(inputs),
        "dimensions": DIMENSIONS,
        "encoding_format": "float",
    }
    return (
        json.dumps({"custom_id": custom_id, "method": "POST", "url": _ENDPOINT, "body": body}) + "\n"
    )  # mutable-ok: JSON request line


def _write_requests(
    path: Path, requests: Sequence[tuple[int, tuple[str, ...]]], manifest: TextIO
) -> tuple[int, int, int]:
    with path.open("w") as handle:
        for number, inputs in requests:
            custom_id = f"r{number:07d}"  # rebind-ok: one request per iteration
            _ = handle.write(_request_line(custom_id, inputs))
            _ = manifest.write(_manifest_line(custom_id, inputs))
    return len(requests), sum(len(inputs) for _, inputs in requests), sum(sum(map(len, i)) for _, i in requests)


def export(chunks: Iterator[str], directory: Path) -> ExportStats:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    groups: Final = batched(enumerate(embedding_batches_stream(chunks)), REQUESTS_PER_FILE)
    with (directory / "manifest.jsonl").open("w") as manifest:
        counts: Final = tuple(
            _write_requests(directory / f"requests-{file_no:04d}.jsonl", group, manifest)
            for file_no, group in enumerate(groups)
        )
    return ExportStats(
        files=len(counts),
        requests=sum(count[0] for count in counts),
        chunks=sum(count[1] for count in counts),
        chars=sum(count[2] for count in counts),
    )


def embedding_batches_stream(chunks: Iterator[str]) -> Iterator[tuple[str, ...]]:
    """embedding_batches over a stream, a bounded window at a time so the whole export never sits in memory"""
    window: tuple[str, ...] = ()  # rebind-ok: refilled as the stream is consumed
    for piece in chunks:
        window = (*window, piece)  # rebind-ok: grows until it can be split
        if len(window) >= 4096:
            batches = tuple(embedding_batches(window))  # rebind-ok: one window per iteration
            yield from batches[:-1]
            window = batches[-1]  # rebind-ok: the unfinished last batch carries over
    if window:
        yield from embedding_batches(window)


class _Batch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    status: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    request_counts: Mapping[str, int] | None = None


class _UploadedFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str


@dataclass(frozen=True, slots=True)
class BatchClient:
    client: httpx.Client
    api_key: str

    def _headers(self) -> Mapping[str, str]:
        return MappingProxyType({"Authorization": f"Bearer {self.api_key}"})

    def upload(self, path: Path) -> str:
        with path.open("rb") as handle:
            response: Final = self.client.post(
                f"{DEEPINFRA_BASE_URL}/openai/files",
                headers=self._headers(),
                data=MappingProxyType({"purpose": "batch"}),
                files=MappingProxyType({"file": (path.name, handle, "application/jsonl")}),
            )
        _ = response.raise_for_status()
        return _UploadedFile.model_validate_json(response.content).id

    def create(self, input_file_id: str) -> _Batch:
        response: Final = self.client.post(
            f"{DEEPINFRA_BASE_URL}/openai/batches",
            headers=MappingProxyType({**self._headers(), "Content-Type": "application/json"}),
            content=json.dumps(
                {"input_file_id": input_file_id, "endpoint": _ENDPOINT, "completion_window": "24h"}
            ),  # mutable-ok: JSON
        )
        _ = response.raise_for_status()
        return _Batch.model_validate_json(response.content)

    def get(self, batch_id: str) -> _Batch:
        response: Final = self.client.get(f"{DEEPINFRA_BASE_URL}/openai/batches/{batch_id}", headers=self._headers())
        _ = response.raise_for_status()
        return _Batch.model_validate_json(response.content)

    def download(self, file_id: str, path: Path) -> None:
        with (
            self.client.stream(
                "GET", f"{DEEPINFRA_BASE_URL}/openai/files/{file_id}/content", headers=self._headers()
            ) as response,
            path.open("wb") as handle,
        ):
            _ = response.raise_for_status()
            for block in response.iter_bytes():
                _ = handle.write(block)


class _ResultRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    index: int
    embedding: tuple[float, ...]


class _ResultBody(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_ResultRow, ...] = ()


class _ResultResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    status_code: int
    body: _ResultBody | None = None


class _ResultLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    custom_id: str
    response: _ResultResponse | None = None


class _ManifestLine(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    custom_id: str
    digests: tuple[str, ...]


def load_vectors(manifest: Path, results: Sequence[Path]) -> Mapping[str, bytes]:
    """Chunk digest to half-precision vector for every successful request in `results`"""
    digests: Final = MappingProxyType(
        {line.custom_id: line.digests for line in map(_ManifestLine.model_validate_json, manifest.open())}
    )
    vectors: Final[dict[str, bytes]] = {}  # mutable-ok: filled from result files too large to build in one pass
    for path in results:
        for raw in path.open():
            line = _ResultLine.model_validate_json(raw)  # rebind-ok: one result per iteration
            if line.response is None or line.response.status_code != 200 or line.response.body is None:
                continue
            names = digests.get(line.custom_id, ())  # rebind-ok: one result per iteration
            for row in line.response.body.data:
                if row.index < len(names) and len(row.embedding) == DIMENSIONS:
                    vectors[names[row.index]] = _HALF.pack(*row.embedding)
    return MappingProxyType(vectors)


def _unpack(stored: bytes) -> tuple[float, ...]:
    return _HALF_TUPLE.validate_python(_HALF.unpack(stored))


@dataclass(frozen=True, slots=True)
class LookupEmbedder:
    """Answers from batch results, and embeds live whatever the batch did not return"""

    vectors: Mapping[str, bytes]
    fallback: Embedder

    def __call__(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        missing: Final = tuple(text for text in texts if chunk_digest(text) not in self.vectors)
        live: Final[Mapping[str, tuple[float, ...]]] = MappingProxyType(
            dict(zip(missing, self.fallback(missing), strict=True)) if missing else {}  # mutable-ok: frozen at once
        )
        return tuple(
            _unpack(stored) if (stored := self.vectors.get(chunk_digest(text))) is not None else live[text]
            for text in texts
        )


def run_import(conn: psycopg.Connection, embed: Embedder) -> IndexStats:
    totals = IndexStats(items=0, texts=0, chunks=0, chars=0)  # rebind-ok: running totals across passes
    while (stats := index_pass(conn, embed, dry_run=False, max_chars=None)).items:
        totals = IndexStats(  # rebind-ok: running totals across passes
            totals.items + stats.items,
            totals.texts + stats.texts,
            totals.chunks + stats.chunks,
            totals.chars + stats.chars,
        )
        sys.stdout.write(f"linked {totals.items} items, {totals.chunks} chunks\n")
        _ = sys.stdout.flush()
    return totals


class _Args(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    command: Literal["export", "submit", "status", "import"]
    directory: Path
    database_url: str | None = None


def _client() -> BatchClient:
    key: Final = os.environ.get("DEEPINFRA_API_KEY")
    if not key:
        raise SystemExit("DEEPINFRA_API_KEY is required")
    return BatchClient(client=httpx.Client(timeout=600), api_key=key)


def _batches_file(directory: Path) -> Path:
    return directory / "batches.json"


def _submit(directory: Path) -> int:
    client: Final = _client()
    batch_ids: Final = tuple(
        client.create(client.upload(path)).id for path in sorted(directory.glob("requests-*.jsonl"))
    )
    _ = _batches_file(directory).write_text(json.dumps(batch_ids))
    sys.stdout.write("".join(f"submitted {batch_id}\n" for batch_id in batch_ids))
    return 0


def _batch_ids(directory: Path) -> tuple[str, ...]:
    return _BATCH_IDS.validate_json(_batches_file(directory).read_text())


def _status(directory: Path) -> int:
    client: Final = _client()
    batches: Final = tuple(client.get(batch_id) for batch_id in _batch_ids(directory))
    sys.stdout.write("".join(f"{batch.id} {batch.status} {batch.request_counts}\n" for batch in batches))
    return 0 if all(batch.status == "completed" for batch in batches) else 1


def _download(directory: Path) -> tuple[Path, ...]:
    client: Final = _client()
    batches: Final = tuple(client.get(batch_id) for batch_id in _batch_ids(directory))
    unfinished: Final = tuple(batch.id for batch in batches if batch.status != "completed")
    if unfinished:
        raise SystemExit(f"not completed yet: {', '.join(unfinished)}")
    paths: Final = tuple(directory / f"results-{batch.id}.jsonl" for batch in batches)
    for batch, path in zip(batches, paths, strict=True):
        if batch.output_file_id and not path.exists():
            client.download(batch.output_file_id, path)
    return tuple(path for path in paths if path.exists())


def _parser() -> argparse.ArgumentParser:
    parser: Final = argparse.ArgumentParser(
        prog="python -m litellm.proxy.memory_vectors_batch", description=(__doc__ or "").split("\n")[0]
    )
    _ = parser.add_argument("--database-url", help="defaults to $DATABASE_URL")
    _ = parser.add_argument("command", choices=("export", "submit", "status", "import"))
    _ = parser.add_argument("directory", type=Path)
    return parser


def _connect(args: _Args) -> psycopg.Connection:
    import psycopg

    url: Final = args.database_url or os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL (or --database-url) is required")
    return psycopg.connect(psycopg_url(url), autocommit=True)


def main() -> int:
    args: Final = _Args.model_validate(vars(_parser().parse_args()))
    if args.command == "submit":
        return _submit(args.directory)
    if args.command == "status":
        return _status(args.directory)
    with _connect(args) as conn:
        if args.command == "export":
            stats: Final = export(missing_chunks(conn), args.directory)
            sys.stdout.write(
                f"{stats.files} files, {stats.requests} requests, {stats.chunks} chunks, {stats.chars} chars\n"
            )
            return 0
        from litellm.proxy.memory_vectors import DeepInfra

        vectors: Final = load_vectors(args.directory / "manifest.jsonl", _download(args.directory))
        sys.stdout.write(f"loaded {len(vectors)} vectors\n")
        live: Final = DeepInfra(client=httpx.Client(timeout=120), api_key=_client().api_key)
        totals: Final = run_import(conn, LookupEmbedder(vectors=vectors, fallback=live.embed))
        sys.stdout.write(f"done: {totals.items} items, {totals.chunks} chunks\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
