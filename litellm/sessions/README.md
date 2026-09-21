# Session manager

## Scope

A gateway-owned transcript for the Responses API, and the things that need one: durable sessions, an upstream
cache key per session, compaction timed to cache expiry, and imports of existing harness histories.

**This is not a new protocol.** There is no new wire format, header or transport. Clients keep using the
Responses API exactly as documented, including `previous_response_id`, and every request that the manager does
not hold falls through to LiteLLM's existing behaviour. An earlier proposal for a persistent-turn protocol
("your move") is retired; only the findings that motivated this work survive, in `~/Projects/your-move`.

## What LiteLLM already does, and why this is not that

Checked against LiteLLM's Responses API documentation and this tree.

| LiteLLM already provides | Where | How the session manager relates |
|---|---|---|
| `previous_response_id` continuation | provider-native, or rebuilt from spend logs by `ResponsesSessionHandler` | Used as-is. The manager answers first only for sessions it holds; it then clears `previous_response_id`, so `async_responses_api_session_handler` (which runs only for a non-empty id) cannot also rebuild. Unheld ids fall through untouched. |
| History in spend logs / cold storage (`store_prompts_in_spend_logs`, `cold_storage_custom_logger`) | spend tracking | Kept for logging and cost. Not a replay source here: it has no item order guarantees, no ownership, no versions. The manager's tables (`LiteLLM_ConversationSession/Item/Response`) exist for replay and rebasing, which logs cannot serve. |
| Deployment affinity for follow-ups (`responses_api_deployment_check`, `session_affinity`) | `router_utils/pre_call_checks/deployment_affinity_check.py` | Left in charge. The router pins the deployment before `litellm.aresponses` runs, so clearing `previous_response_id` inside it does not affect the pin. Compaction reuses the turn's deployment id for the same reason. |
| Routing to a deployment that already cached a prefix | `pre_call_checks/prompt_caching_deployment_check.py` | Different layer: that picks a deployment, `cache_affinity` sets `prompt_cache_key` so the provider's own cache stays warm across turns of one session. |
| Per-model minimum cacheable prefix | `litellm.utils.get_prompt_cache_min_tokens` | Used directly. A local 1024-token constant was retired in favour of it. |
| Anthropic prompt-cache prediction: `cache_control` parsing, 5m/1h TTLs, observed-hit records, `/cache/prediction` | `llms/anthropic/prompt_cache_prediction.py`, `proxy/hooks/prompt_cache_prediction.py` | Owns Anthropic's *declared* TTL, for `/v1/messages`. `cache_clock` answers a different question on the Responses path — how long a prefix *actually* stays warm for a provider and model that declares nothing — and no longer parses `cache_control` itself. |
| Prompt compression guardrails (`headroom`, `compresr`, `typesafe`) | `proxy/guardrails/` | Per-request prompt shrinking by a third party. Compaction instead rewrites the stored session once, while its cache is warm, and the result is what later turns replay. |
| Session ids from client headers (`litellm_session_id`, e.g. opencode's `x-session-id`, Codex's `session-id`) | `proxy/litellm_pre_call_utils.py` | Adopted, not duplicated. It is recorded on the session (`client_session_id`, persisted in the session row's metadata) and it, not a gateway-invented id, keys the upstream cache: `prompt_cache_key = affinity_key(owner, litellm_session_id)`. The manager's own id stays opaque; see "Session identity" below. |
| WebSocket per-connection history | `ManagedResponsesWebSocketHandler` | Still the fallback. With the manager enabled, the connection's own transcript is bypassed so a session survives reconnects and restarts. |
| Response retrieve/delete (`GET`/`DELETE /v1/responses/{id}`) | provider passthrough | Untouched. |

## What is added here

- **`manager.py`** – the live transcript, addressed by the response ids the gateway returned; ownership checks;
  fork detection; LRU eviction; a background writer to the store.
- **`session.py`** – a session's items and the projection replayed upstream (assistant text, encrypted reasoning).
- **`responses.py`** – the Responses API adapter: begin/commit a turn, `prompt_cache_key` affinity.
- **`cache_clock.py`** – learned warm-window estimate per provider and model.
- **`compaction*.py`** – compaction just before predicted expiry, its policies and size thresholds
  (see `COMPACTION.md`).
- **`importers/`** – opencode, openclaw and Claude Code histories as neutral Responses items
  (see `../proxy/session_import.py`). The Claude Code reader is also what keeps the store level with the
  `claude_cli` provider, whose conversation lives in Claude Code rather than in anything the proxy saw.
- **`MEMORY_DESIGN.md`** – direction only; nothing built.

Proxy wiring lives in `litellm/proxy/sessions.py`, storage in
`litellm/repositories/conversation_session_repository.py`.

## Session identity

There is one client-facing session identity, LiteLLM's `litellm_session_id`, and the gateway does not invent a
second one for anything the client or the provider can see:

- **Upstream cache key**: `sess_<sha256(owner + "\0" + litellm_session_id)[:32]>`. Hashed because the value is
  client-chosen — two API keys must never share a cache key — and so the client's own id never travels upstream.
  It is stable across gateway restarts, so a session that outlives the process still lands on a warm cache.
- **Correlation**: the session row records `litellm_session_id` in its metadata, so stored transcripts line up
  with spend logs and traces.
- **Internal id**: resolution stays keyed on the response ids the gateway returned, and a turn that names one
  is answered from that alone. A turn that names only a session lands at
  `cs_<sha256("session" + owner + litellm_session_id)[:32]>` — derived, so the next turn and a restarted
  gateway find the same row with no index to keep, and owner-hashed, so one key cannot guess its way to
  another's. A client-chosen id still never *addresses* stored history on its own: see below.

## Clients that keep their own history

A harness with its own checkpointer — LangGraph, and most agent frameworks — sends the whole conversation
every turn and no `previous_response_id`. Taken at face value every turn is a new conversation, and the store
fills with one session per turn, each a superset of the last: measured, three turns of one LangGraph thread
produced three sessions of 2, 4 and 6 items.

Such a turn is accepted as a continuation only when what arrived **reproduces what is stored**. The arriving
input is compared against the session's own transcript on what was said rather than how it was wrapped (roles
and text, tool calls and their results; reasoning is left out, since it is dropped on replay unless sealed and
a harness need not echo it). If the stored history is a prefix of what arrived, the remainder is the new turn
and the rest is replayed from the gateway's own copy. If it is not, the turn begins a conversation of its own
and the stored one is untouched.

That keeps the property the identity rules are there for. A guessable id reaches nothing by itself: the id is
namespaced by owner, and extending a transcript requires already being able to reproduce it. What it buys is
that a thread is one session — one row, one set of response ids, and a history long enough for compaction to
be worth doing.

Nothing changes for a client that sends `previous_response_id`, or for one that sends no session id at all.

## `store: false`

Harnesses that keep their own history — opencode, Codex — send `store: false` on every turn. The flag asks the
*provider* not to keep the response. By default the manager reads it the same way and records nothing, which
on a gateway that is its owner's own session store means none of those harnesses' turns are ever recorded.

`session_manager.record_store_false: true` records them anyway, with one rule: **a `store: false` turn is
recorded, never rewritten.** The client owns that conversation's state, so the provider receives the input
exactly as the client sent it — no replayed history, and `previous_response_id` passes through untouched. The
gateway only keeps the record, continuing the client's session by the rule in "Clients that keep their own
history". The one thing still added is `prompt_cache_key`, and only when the client sent none: a routing hint
that keeps the provider's cache warm, not content.

Compaction and recording-only turns don't combine: a compacted transcript no longer reproduces the history
the client resends, so the next turn would start a new session. Leave compaction off where most traffic is
`store: false`.

## Open

- Compaction calls are attributed to the session's key in metadata but may not appear in per-key spend tracking.
