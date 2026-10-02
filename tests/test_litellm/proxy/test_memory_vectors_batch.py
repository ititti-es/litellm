import json
from pathlib import Path
from typing import Final

import psycopg

from litellm.proxy.memory_vectors import DIMENSIONS, EMBED_BATCH_SIZE, embedding_batches
from litellm.proxy.memory_vectors_batch import (
    _HALF,
    LookupEmbedder,
    chunk_digest,
    embedding_batches_stream,
    export,
    load_vectors,
    missing_chunks,
    run_import,
)
from tests.test_litellm.proxy.test_memory_vectors import (
    _bag_of_words,
    _Embedder,
    _vectors_postgresql,  # noqa: F401  # fixture, found by pytest through this module's namespace
    _vectors_postgresql_proc,  # noqa: F401  # fixture, found by pytest through this module's namespace
    store_fixture,  # noqa: F401  # fixture registered as "store"
)


def _requests(directory: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (line["custom_id"], tuple(line["body"]["input"]))
        for path in sorted(directory.glob("requests-*.jsonl"))
        for line in map(json.loads, path.open())
    )


def _write_results(directory: Path, failed: frozenset[str] = frozenset()) -> Path:
    path: Final = directory / "results-test.jsonl"
    _ = path.write_text(
        "".join(
            json.dumps(
                {"custom_id": custom_id, "response": {"status_code": 500, "body": None}}
                if custom_id in failed
                else {
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "data": tuple({"index": n, "embedding": _bag_of_words(t)} for n, t in enumerate(inputs))
                        },
                    },
                }
            )
            + "\n"
            for custom_id, inputs in _requests(directory)
        )
    )
    return path


def test_export_lists_each_missing_chunk_once_and_the_manifest_names_every_input(
    store: psycopg.Connection, tmp_path: Path
) -> None:
    stats: Final = export(missing_chunks(store), tmp_path)
    inputs: Final = tuple(text for _, batch in _requests(tmp_path) for text in batch)
    assert sorted(inputs) == sorted(
        (
            "user: my tomato seedlings are wilting in the greenhouse",
            "assistant: water the seedlings less and open the greenhouse vents",
            "user: postgres vacuum full locked the sessions table",
        )
    )
    assert (stats.chunks, stats.requests) == (3, 1)
    manifest: Final = tuple(map(json.loads, (tmp_path / "manifest.jsonl").open()))
    assert tuple(digest for line in manifest for digest in line["digests"]) == tuple(map(chunk_digest, inputs))


def test_export_skips_texts_whose_chunks_are_already_stored(store: psycopg.Connection, tmp_path: Path) -> None:
    _ = run_import(store, _Embedder())
    store.execute(
        """INSERT INTO "LiteLLM_ConversationItem" (session_id, version, seq, content) VALUES
           ('ses_db', 0, 1, %s), ('ses_db', 0, 2, %s)""",
        (
            json.dumps(
                {"type": "message", "role": "user", "content": "postgres vacuum full locked the sessions table"}
            ),
            json.dumps({"type": "message", "role": "user", "content": "a brand new question"}),
        ),
    )
    _ = export(missing_chunks(store), tmp_path)
    assert tuple(text for _, batch in _requests(tmp_path) for text in batch) == ("user: a brand new question",)


def test_import_links_every_item_from_batch_results_without_live_calls(
    store: psycopg.Connection, tmp_path: Path
) -> None:
    _ = export(missing_chunks(store), tmp_path)
    vectors: Final = load_vectors(tmp_path / "manifest.jsonl", (_write_results(tmp_path),))
    live: Final = _Embedder()
    totals: Final = run_import(store, LookupEmbedder(vectors=vectors, fallback=live))
    assert totals.items == 5
    assert live.texts == ()
    assert store.execute('SELECT count(*) FROM "LiteLLM_ConversationItemText"').fetchone() == (5,)
    distance: Final = store.execute(
        """SELECT embedding <=> %s::halfvec FROM "LiteLLM_ConversationChunk" WHERE content LIKE 'assistant:%%'""",
        (
            "["
            + ",".join(map(str, _bag_of_words("assistant: water the seedlings less and open the greenhouse vents")))
            + "]",
        ),
    ).fetchone()
    assert distance is not None and distance[0] < 1e-3


def test_import_embeds_live_only_what_a_failed_request_did_not_return(
    store: psycopg.Connection, tmp_path: Path
) -> None:
    _ = export(missing_chunks(store), tmp_path)
    failed: Final = frozenset(custom_id for custom_id, _ in _requests(tmp_path))
    vectors: Final = load_vectors(tmp_path / "manifest.jsonl", (_write_results(tmp_path, failed),))
    live: Final = _Embedder()
    _ = run_import(store, LookupEmbedder(vectors=vectors, fallback=live))
    assert len(live.texts) == 3
    assert store.execute('SELECT count(*) FROM "LiteLLM_ConversationItemText"').fetchone() == (5,)


def test_streamed_batches_match_the_in_memory_batching() -> None:
    texts: Final = tuple(f"chunk {n} " + "x" * (n % 3000) for n in range(EMBED_BATCH_SIZE * 90))
    streamed: Final = tuple(embedding_batches_stream(iter(texts)))
    assert tuple(text for batch in streamed for text in batch) == texts
    assert all(batch for batch in streamed)
    assert max(map(len, streamed)) <= EMBED_BATCH_SIZE
    assert len(streamed) <= len(tuple(embedding_batches(texts))) + len(texts) // 4096


def test_lookup_embedder_prefers_stored_vectors() -> None:
    stored_text: Final = "stored"
    live: Final = _Embedder()
    stored_vector: Final = (1.0,) + (0.0,) * (DIMENSIONS - 1)
    embedder: Final = LookupEmbedder(vectors={chunk_digest(stored_text): _HALF.pack(*stored_vector)}, fallback=live)
    result: Final = embedder((stored_text, "fresh"))
    assert result[0] == stored_vector
    assert live.texts == ("fresh",)
