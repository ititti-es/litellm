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
- **Internal id**: the manager keeps minting an opaque id, and resolution stays keyed on the response ids the
  gateway returned. A client-chosen id must not address stored history: ids are guessable, and a turn arriving
  with no `previous_response_id` would otherwise overwrite a live session's transcript (or, for a harness that
  resends its whole history every turn, write a new version per turn). Ownership is checked on every resolve
  regardless.

## Open

- Compaction calls are attributed to the session's key in metadata but may not appear in per-key spend tracking.
