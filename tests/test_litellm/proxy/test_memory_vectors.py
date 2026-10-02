import hashlib
import json
import math
import re
from collections.abc import Sequence
from typing import Final

import httpx
import psycopg
import pytest
from pytest_postgresql import factories

from litellm.proxy.memory_vectors import (
    CHUNK_OVERLAP,
    DIMENSIONS,
    EMBED_BATCH_CHARS,
    EMBED_BATCH_SIZE,
    RETRY_ATTEMPTS,
    SCHEMA_SQL,
    DeepInfra,
    chunk,
    embedding_batches,
    index_pass,
    labeled_text,
    reciprocal_rank_fusion,
    search,
)

_vectors_postgresql_proc: Final = factories.postgresql_proc()
_vectors_postgresql: Final = factories.postgresql("_vectors_postgresql_proc")

_STORE_SQL: Final = """
CREATE TABLE "LiteLLM_ConversationSession" (
    session_id TEXT PRIMARY KEY, owner TEXT NOT NULL, version INTEGER NOT NULL,
    title TEXT, source TEXT, updated_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE "LiteLLM_ConversationItem" (
    session_id TEXT NOT NULL REFERENCES "LiteLLM_ConversationSession" ON DELETE CASCADE,
    version INTEGER NOT NULL, seq INTEGER NOT NULL, content JSONB NOT NULL,
    PRIMARY KEY (session_id, version, seq)
);
"""


def _words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z]+", text.lower()))


def _bag_of_words(text: str) -> tuple[float, ...]:
    slots: Final = tuple(int(hashlib.md5(word.encode()).hexdigest(), 16) % DIMENSIONS for word in _words(text))
    raw: Final = tuple(float(slots.count(index)) for index in range(DIMENSIONS))
    norm: Final = math.sqrt(sum(value * value for value in raw)) or 1.0
    return tuple(value / norm for value in raw)


class _Embedder:
    def __init__(self) -> None:
        self.texts: tuple[str, ...] = ()

    def __call__(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.texts = (*self.texts, *texts)
        return tuple(_bag_of_words(text) for text in texts)


def _overlap_rerank(query: str, documents: Sequence[str]) -> tuple[float, ...]:
    wanted: Final = frozenset(_words(query))
    return tuple(len(wanted & frozenset(_words(document))) / (len(wanted) or 1) for document in documents)


def _failing_embedder(texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
    raise RuntimeError("provider down")


def _message(role: str, text: str) -> str:
    return json.dumps({"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]})


@pytest.fixture
def store(_vectors_postgresql: psycopg.Connection) -> psycopg.Connection:
    conn: Final = _vectors_postgresql
    conn.autocommit = True
    conn.execute(_STORE_SQL)
    conn.execute(SCHEMA_SQL)
    conn.execute(
        """INSERT INTO "LiteLLM_ConversationSession" (session_id, owner, version, title) VALUES
           ('ses_garden', 'alice', 1, 'garden'), ('ses_db', 'bob', 0, 'database')"""
    )
    tomato: Final = _message("user", "my tomato seedlings are wilting in the greenhouse")
    rows: Final = (
        ("ses_garden", 0, 0, tomato),
        ("ses_garden", 1, 0, tomato),
        ("ses_garden", 1, 1, json.dumps({"type": "reasoning", "summary": []})),
        ("ses_garden", 1, 2, _message("assistant", "water the seedlings less and open the greenhouse vents")),
        ("ses_db", 0, 0, _message("user", "postgres vacuum full locked the sessions table")),
    )
    with conn.cursor() as cursor:
        cursor.executemany(
            'INSERT INTO "LiteLLM_ConversationItem" (session_id, version, seq, content) VALUES (%s, %s, %s, %s)', rows
        )
    return conn


def test_chunks_cover_the_capped_text_within_size_and_overlap() -> None:
    text: Final = "\n".join(f"line {n} " + "word " * (n % 17) for n in range(900))
    pieces: Final = chunk(text, size=500, overlap=50, cap=20_000)
    assert all(len(piece) <= 500 for piece in pieces)
    assert all(piece in text for piece in pieces)
    assert pieces[0].startswith("line 0")
    assert text[:20_000].rstrip().endswith(pieces[-1])
    covered: Final = frozenset(
        offset for piece in pieces for offset in range(text.find(piece), text.find(piece) + len(piece))
    )
    assert all(offset in covered for offset, char in enumerate(text[:20_000]) if not char.isspace())
    assert len(chunk(text, size=500, overlap=50, cap=1000)) < len(pieces)


def test_chunks_break_at_a_line_rather_than_mid_word() -> None:
    text: Final = "alpha " * 60 + "\n" + "omega " * 60
    first: Final = chunk(text, size=400, overlap=CHUNK_OVERLAP)[0]
    assert first == ("alpha " * 60).strip()


def test_short_text_is_one_chunk_and_blank_text_is_none() -> None:
    assert chunk("hello there") == ("hello there",)
    assert chunk("   ") == ()


def test_labeled_text_names_the_speaker_and_skips_reasoning() -> None:
    assert labeled_text({"type": "reasoning", "summary": []}) is None
    assert (
        labeled_text({"type": "function_call", "name": "grep", "arguments": '{"q": "x"}'})
        == 'tool call grep: {"q": "x"}'
    )
    assert labeled_text({"type": "function_call_output", "output": "a\x00b"}) == "tool output: ab"
    assert (
        labeled_text({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]})
        == "user: hi"
    )


def test_fusion_ranks_a_hit_found_by_both_legs_first() -> None:
    fused: Final = reciprocal_rank_fusion((((b"a", 0), (b"b", 0)), ((b"c", 0), (b"b", 0))))
    assert fused[0] == (b"b", 0)
    assert frozenset(fused) == {(b"a", 0), (b"b", 0), (b"c", 0)}


def test_embedding_batches_keep_order_and_respect_both_limits() -> None:
    texts: Final = tuple("x" * (EMBED_BATCH_CHARS // 3 + n) for n in range(7)) + tuple(
        "y" for _ in range(EMBED_BATCH_SIZE + 5)
    )
    batches: Final = tuple(embedding_batches(texts))
    assert tuple(text for batch in batches for text in batch) == texts
    assert all(batch for batch in batches)
    assert all(len(batch) <= EMBED_BATCH_SIZE for batch in batches)
    assert all(sum(map(len, batch)) <= EMBED_BATCH_CHARS for batch in batches if len(batch) > 1)


def test_index_embeds_each_distinct_text_once_and_links_every_item(store: psycopg.Connection) -> None:
    embedder: Final = _Embedder()
    first: Final = index_pass(store, embedder, dry_run=False, max_chars=None)
    assert (first.items, first.texts) == (5, 3)
    assert sorted(embedder.texts) == sorted(
        (
            "user: my tomato seedlings are wilting in the greenhouse",
            "assistant: water the seedlings less and open the greenhouse vents",
            "user: postgres vacuum full locked the sessions table",
        )
    )
    links: Final = store.execute('SELECT count(*), count(text_sha) FROM "LiteLLM_ConversationItemText"').fetchone()
    assert links == (5, 4)
    again: Final = index_pass(store, _Embedder(), dry_run=False, max_chars=None)
    assert again.items == 0


def test_a_failed_embedding_leaves_items_unlinked_for_the_next_pass(store: psycopg.Connection) -> None:
    with pytest.raises(RuntimeError):
        index_pass(store, _failing_embedder, dry_run=False, max_chars=None)
    assert store.execute('SELECT count(*) FROM "LiteLLM_ConversationItemText"').fetchone() == (0,)
    assert store.execute('SELECT count(*) FROM "LiteLLM_ConversationChunk"').fetchone() == (0,)
    assert index_pass(store, _Embedder(), dry_run=False, max_chars=None).items == 5


def test_dry_run_and_char_budget_embed_nothing(store: psycopg.Connection) -> None:
    embedder: Final = _Embedder()
    preview: Final = index_pass(store, embedder, dry_run=True, max_chars=None)
    over_budget: Final = index_pass(store, embedder, dry_run=False, max_chars=preview.chars - 1)
    assert preview.chars > 0
    assert over_budget == preview
    assert embedder.texts == ()
    assert store.execute('SELECT count(*) FROM "LiteLLM_ConversationItemText"').fetchone() == (0,)


def test_search_finds_the_session_and_respects_the_owner(store: psycopg.Connection) -> None:
    index_pass(store, _Embedder(), dry_run=False, max_chars=None)
    hits: Final = search(store, _Embedder(), _overlap_rerank, "greenhouse vents", owner=None, limit=5)
    assert hits[0].session_id == "ses_garden"
    assert "vents" in hits[0].content
    assert hits[0].seq == 2
    assert all(
        hit.session_id == "ses_db"
        for hit in search(store, _Embedder(), _overlap_rerank, "greenhouse", owner="bob", limit=5)
    )


def test_search_returns_a_text_shared_by_two_versions_once(store: psycopg.Connection) -> None:
    index_pass(store, _Embedder(), dry_run=False, max_chars=None)
    hits: Final = search(store, _Embedder(), _overlap_rerank, "tomato wilting", owner=None, limit=5)
    tomato: Final = next(hit for hit in hits if "tomato" in hit.content)
    assert (tomato.session_id, tomato.seq) == ("ses_garden", 0)
    assert sum(1 for hit in hits if "tomato" in hit.content) == 1


class _Waits:
    def __init__(self) -> None:
        self.seconds: tuple[float, ...] = ()

    def __call__(self, seconds: float) -> None:
        self.seconds = (*self.seconds, seconds)


def _deepinfra(statuses: Sequence[int]) -> tuple[DeepInfra, _Waits]:
    remaining: Final = iter(statuses)
    waits: Final = _Waits()

    def respond(request: httpx.Request) -> httpx.Response:
        status: Final = next(remaining)
        return httpx.Response(status, content=b'{"scores": [0.5]}') if status == 200 else httpx.Response(status)

    client: Final = httpx.Client(transport=httpx.MockTransport(respond))
    return DeepInfra(client=client, api_key="test", sleep=waits), waits


def test_rerank_retries_rate_limits_until_it_succeeds() -> None:
    deepinfra, waits = _deepinfra((429, 503, 200))
    assert deepinfra.rerank("query", ("document",)) == (0.5,)
    assert len(waits.seconds) == 2
    assert waits.seconds[1] > waits.seconds[0]


def test_rerank_gives_up_after_the_last_attempt_and_never_retries_a_client_error() -> None:
    limited, limited_waits = _deepinfra((429,) * RETRY_ATTEMPTS)
    with pytest.raises(httpx.HTTPStatusError):
        _ = limited.rerank("query", ("document",))
    assert len(limited_waits.seconds) == RETRY_ATTEMPTS - 1
    rejected, rejected_waits = _deepinfra((400,))
    with pytest.raises(httpx.HTTPStatusError):
        _ = rejected.rerank("query", ("document",))
    assert rejected_waits.seconds == ()
