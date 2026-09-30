# Session-store operations

## Components and ownership

OpenCode owns its local SQLite history and tool execution. The gateway owns provider routing and a
PostgreSQL transcript projection. PostgreSQL is not a replacement for OpenCode's SQLite database

| Component | Implementation | Durable state |
| --- | --- | --- |
| Responses recording and replay | `sessions/responses.py`, `sessions/manager.py`, `proxy/sessions.py` | Conversation session, item and response tables |
| Transcript persistence | `repositories/conversation_session_repository.py` | Ordered, versioned items with owner and response boundaries |
| Harness imports | `proxy/session_import.py`, `sessions/importers/` | Imported sessions, source IDs and provenance |
| Compaction | `sessions/compaction.py`, `sessions/compaction_policy.py` | Rebased transcript versions and summary metadata |
| Memory backfill and search | `proxy/memory_backfill.py`, `proxy/memory_search.py` | Item `search_text`, segment/session nodes and GIN indexes |
| Subscription credentials | `proxy/oauth_credential_store.py`, `proxy/oauth_cli.py` | PostgreSQL credential records and provider-local fallback files |
| Claude external tools | `llms/claude_cli/tool_bridge.py`, `llms/claude_cli/mcp_sidecar.py` | Process-local relay state; completed history supplied by the client |

Paths in the table are relative to `litellm/`. A recorded session and a Claude bridge process have
different lifetimes. Durable transcript storage does not make a waiting MCP tool invocation durable

## Local deployment layout

The local installation uses these locations. They are deployment conventions, not requirements of the SDK

| Location | Purpose |
| --- | --- |
| `~/Projects/ititti.es/liminal` | Source and development environment |
| `~/.local/lib/liminal/litellm` | Deployed package and its virtual environment |
| `~/.config/litellm/config.yaml` | Model routes, custom providers and session settings |
| `~/.config/litellm/secrets/` | Service credentials; never include contents in diagnostics |
| `~/.config/systemd/user/liminal-litellm.service` | Gateway service, served on loopback port 4000 |
| `~/.config/opencode-v2-fork/opencode/opencode.jsonc` | Client model catalog and agent assignments |
| `~/Projects/internal/sessiondb` | Local database maintenance CLI |
| `~/Projects/internal/liminalctl` | Local gateway maintenance CLI |

The maintenance CLIs are separate projects, not installed by this repository. In this installation,
`sessiondb` runs import/backfill modules from the source virtual environment, while the gateway runs
the deployed virtual environment. Check both before assuming a successful maintenance command proves
that production has the same implementation

## Readiness and database checks

```bash
sessiondb check
liminalctl check
curl --fail --max-time 5 http://127.0.0.1:4000/health/liveliness
sessiondb stats
```

A systemd `active` state alone does not prove that application startup finished. `liminalctl check`
checks the listener through the authenticated model endpoint and compares configured routes with the
client catalog. Embedding, rerank, speech and transcription models are endpoint-only, not chat models

Database checks should verify connectivity, unfinished migrations, the conversation tables, the
`search_text` column, the memory-node table and both full-text GIN indexes. The memory migration lives
in `litellm-proxy-extras/litellm_proxy_extras/migrations/20260922021208_add_memory_summary_nodes/`
and changes schema only. Backfills run separately from proxy startup

Missing search text, unimported sessions and stale imports are data-completeness warnings. They do
not by themselves mean routing is unavailable. Counts change with traffic; compare trends rather
than treating a fixed count as a health invariant

## Imports and backfills

```bash
sessiondb import
sessiondb backfill
```

Both commands default to dry-run. Inspect their proposed counts before applying changes. Import
dry-runs must read the existing projection to distinguish unchanged, appended and rewritten history,
and must not write to it

```bash
sessiondb backup --postgres
sessiondb import --apply
sessiondb backfill --apply
```

Applying a backfill can touch a large number of rows. Schedule it separately from deployment and
watch database space, WAL and latency. Backfill fills NULL search text and upserts memory nodes by
deterministic IDs. Segment summaries come from existing compaction records; session summaries concatenate
them, without paid model calls. Project/scope summaries, vector retrieval and live expansion are not implemented

Search commands can print conversation snippets. Do not run them as generic diagnostics or paste
their output into public issues. Use counts and schema metadata for routine health checks

## Provider recovery

ChatGPT requests and gateway startup never start device login automatically. Missing or expired
credentials produce an authentication error. Use the explicit login CLI with the service's credential
environment loaded locally, then push the refreshed credentials to the configured database

```bash
~/.local/lib/liminal/litellm/.venv/bin/litellm chatgpt login --force
~/.local/lib/liminal/litellm/.venv/bin/litellm chatgpt push
liminalctl reload
```

Do not paste the environment file, tokens or authorization headers into logs. A failed provider login
must not keep unrelated model routes from becoming ready

Claude uses Claude Code subscription billing, a strict generated MCP configuration and no built-in
gateway-host tools. OpenCode receives structured function calls, executes its own tools and returns
results with the same call IDs. The bridge does not execute those tools itself

Completed call/result history can start a fresh process after expiry or gateway restart. Changing
model, instructions, tool definitions or credential configuration rebuilds the process from supplied
history. Recovery requires complete call/result pairs. Outstanding calls without results fail with
HTTP 409; never synthesize a result or retry an external side effect solely to clear that error

Completed tool results followed by new steering input rebuild from the supplied history so the new
input is not dropped. Replaying a completed result request against an idle live bridge returns
HTTP 409. Completed turns rebuild from client history on their next request, trading process and cache
reuse for safe handling of edited histories

OpenCode's own `x-session-id` identifies the conversation. Its `x-parent-session-id` remains trace
chain metadata, so parallel children do not share the parent's transcript or waiting relay process

Recovery does not provide an exactly-once transaction spanning an external tool and PostgreSQL.
Preserve tool results in the client and inspect uncertain side effects before retrying. Concurrent
requests sharing one client session must be serialized. Active relay state is process-local, so a
multi-worker deployment needs affinity for each waiting tool round

The relay registry has a shared process limit. At capacity, idle processes without pending calls are
evicted first. If all slots have active turns or unresolved calls, a new session receives HTTP 503

Only function tools with absent/default or `auto` tool choice are supported by the Claude bridge.
Image input is unsupported and rejected explicitly. Do not advertise image capability for these routes

## Deployment safety

Before deployment, inspect both worktrees and record the source commit, deployed commit and local
modifications. Back up changed files and PostgreSQL. Preserve locally modified model catalogs and
provider-specific features on deployment-only branches; a wholesale source copy can remove them

Schema changes must be installed through the normal migration mechanism and the Prisma client must
match the schema. Never combine unbounded row rewrites with startup migrations. A file-only hotfix
does not update the deployed checkout's Git HEAD, so record the exact hotfixed files as well as the commit

Restart only the gateway when deploying its implementation. `liminalctl reload` waits for readiness
and then checks the model catalog. Confirm both stages; a final catalog error can occur after the
server has already restarted successfully. OpenCode need not restart for a gateway-only fix

## Verification

Focused regression suites can run without exposing real histories

```bash
uv run --no-sync pytest -q tests/test_litellm/sessions
uv run --no-sync pytest -q tests/test_litellm/proxy/test_sessions.py \
  tests/test_litellm/proxy/test_session_import.py \
  tests/test_litellm/proxy/test_oauth_credential_store.py \
  tests/test_litellm/proxy/test_oauth_cli.py \
  tests/test_litellm/proxy/test_claude_cli_sync.py
uv run --no-sync pytest -q tests/test_litellm/llms/claude_cli \
  tests/test_litellm/llms/chatgpt/test_chatgpt_authenticator.py
make check
```

Read the complete `make check` log path printed by that command. Do not lower lint/type budgets to
clear a failure. Unit tests are necessary but do not prove PostgreSQL round trips, provider login,
real model availability or the OpenCode tool surface

Use real provider probes for route readiness; these spend subscription quota or provider credit

```bash
liminalctl probe gpt-6.1-sol --timeout 90
liminalctl probe gpt-6-luna --timeout 90
liminalctl probe claude-opus-5-5 --timeout 90
liminalctl probe claude-sonnet-5-5 --timeout 90
```

Tool verification must additionally cover a real call/result round trip, multiple tool rounds,
stream completion, existing completed history, changed instructions/tools, model switching and
recovery after process loss. Use synthetic markers, not actual conversation text. Inspect the
OpenCode surface as well as the gateway before claiming end-to-end operation
