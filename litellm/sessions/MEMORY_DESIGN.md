# Hierarchical memory over the session store (design, not built)

Status: direction only. Nothing here is implemented; the components it depends on (cache-expiry prediction,
pre-expiry compaction) are being built first and should keep this design reachable.

## Goal

The session store holds every session from every harness (opencode, openclaw, new harnesses) in one harness-neutral
form. That makes it the basis for a personal or team memory system:

- **Scopes** are defined over the store: a person, a team, a project.
- **The total context is one monolith**, organised as a hierarchy. Nothing is loaded wholesale; a search decides which
  sections are expanded, and to what depth, within a token budget. Everything else stays folded to a one-line
  summary with an address that can be expanded later.

## The hierarchy

```
scope        (person / team / org)
 └ project   (repo or cwd, from session metadata)
    └ session   (any harness; provenance per turn)
       └ segment   (the span one compaction covered)
          └ items     (lossless leaves: what the store holds today)
```

Leaves are the stored Responses items plus `meta` (source record, kind, provenance, tool class). Every level above
the leaves is a **summary node**. Because leaves are never discarded, any summary can be checked against, or
expanded back into, its source; summary drift cannot compound unchecked.

## Compaction produces the internal nodes

Compaction and memory are the same mechanism seen from two sides:

- Pre-expiry compaction (run while the prompt cache is still warm, just before its predicted expiry) summarises a
  span of a session. Its output is a summary node, not throwaway text spliced into one transcript.
- Harness compaction summaries already in the store (opencode `summary` messages, openclaw `compaction` entries,
  `meta.kind = compaction_summary`) are free internal nodes over the turns they replaced.
- Summaries of summaries form the session, project and scope levels.
- Several compaction policies (harness-emulating or custom, `LiteLLM_CompactionPolicy`) may each summarise the same
  span; a node records which policy and model produced it.

## Summary node record (sketch)

| field | purpose |
|---|---|
| `node_id` | stable address, used by `expand` |
| `parent_id` | position in the hierarchy |
| `scope` | who may see it (see Scopes) |
| `span` | `(session_id, version, first_seq, last_seq)` for segment nodes; child node ids above that |
| `level` | segment, session, project, scope |
| `summary` | the text |
| `policy`, `model` | what produced it |
| `tokens` | cost to include when folded vs expanded |
| `embedding` | pgvector, for search |
| `created_at`, `source_versions` | invalidation when a session is rebased to a new version |

## Search and rendering

- **Retrieval** is hybrid and stays in Postgres: full-text (`tsvector`) over summaries and leaf text, plus pgvector
  similarity. Provenance and the canonical tool classes allow filtering or ranking by harness, model or toolset.
- **Expansion** is a budgeted choice: starting from the folded root, repeatedly expand the node with the highest
  relevance gained per token until the budget is spent. The result is a view: expanded nodes in full, the rest as
  one-liners carrying their `node_id`.

## Interaction with the prompt cache

The rendered view is the start of the prompt, so rewriting it mid-session would invalidate the cache — the class of
bug the session manager exists to remove. Therefore:

- **During a session, expansion is an append**: an `expand(node_id)` tool whose result is the node's content. The
  prefix never changes.
- **The view is re-folded only at compaction time**, when the prefix is being rewritten anyway, just before the
  cache would expire. One timing rule serves both compaction and memory.

## Scopes

Today a session's owner is one virtual-key hash. Memory needs a real scope model (person, team, project) with
membership. Rules:

- **A summary is never wider than the narrowest scope of its inputs.** A team-scope summary built from a private
  session leaks it. Summaries are built within one scope.
- **Promotion to a wider scope is explicit**, and passes through redaction: transcripts contain tool output, which
  can contain secrets (keys, tokens, env dumps).
- Continuation rights (owner-only today) and read rights for memory are separate permissions.

## Order of work

1. Cache-expiry prediction, then pre-expiry compaction whose output is stored as summary nodes (not built as memory
   yet, but in the shape above).
2. The node table and hierarchy across sessions and projects; backfill summaries for imported sessions.
3. Search and the folded view with `expand`.
4. Scopes, promotion and redaction.

## Open questions

- Project identity across harnesses: cwd, git remote, or explicit tagging.
- Whether session-level and project-level nodes are refreshed incrementally or rebuilt on a schedule.
- Which embedding model, and whether embeddings are re-computed when the model changes.
- How a harness adapter presents folded memory in each harness's native toolset.
