# Subscription logins

Some providers will bill inference to a personal subscription — Claude Pro/Max, a SuperGrok plan, ChatGPT
Plus/Pro through Codex — instead of a pay-per-token API key. Each does it with an OAuth token that expires in
hours and has to be refreshed, so it cannot simply be pasted into `api_key` and left alone.

`litellm <provider> login` performs that login once, and the provider's authenticator refreshes the token
from then on.

## Commands

```sh
litellm anthropic-oauth login     # Claude Pro/Max
litellm xai-oauth login           # SuperGrok
litellm chatgpt login             # ChatGPT Plus/Pro (Codex)

litellm <provider> status         # logged in? when does the token expire?
litellm <provider> logout         # forget the stored credentials
litellm <provider> push           # copy the local credentials into the database
litellm <provider> pull           # copy them back out
```

Flags on `login`:

| flag | effect |
|---|---|
| `--force` | ignore a stored token and log in again |
| `--manual` | print the URL and prompt for the code, instead of waiting on a loopback redirect |
| `--code VALUE` | finish a login started earlier, from a pasted code or redirect URL |
| `--browser` | force the loopback flow even in a session detected as remote |
| `--no-browser` | print the authorization URL rather than opening a browser, but still wait on the loopback |

## Logging in over SSH

Anthropic and xAI receive their authorization code on a loopback port. Over SSH that port is on the
*client's* machine, where nothing is listening, so the redirect never arrives and `webbrowser.open` can
report success having done nothing.

`login` therefore detects a remote session — `SSH_CONNECTION`, `SSH_TTY`, `SSH_CLIENT`, or a Linux host
with neither `DISPLAY` nor `WAYLAND_DISPLAY` — and asks for the code instead.

Where there is a terminal, it prompts. Where there is not — a pipe, a script, a non-interactive remote
command — it prints the URL and stops, because `click.prompt` would only hit EOF and abort:

```
$ litellm anthropic-oauth login
1. Open this URL in a browser on any machine:

   https://claude.ai/oauth/authorize?code=true&client_id=...

2. Approve the request there.
3. The browser then lands on a 127.0.0.1 page that FAILS TO LOAD. That is the success case.
4. Copy that failed page's address (not the URL above) -- the code is in it.

Then finish the login with:
  litellm anthropic-oauth login --code '<the code>'

The link is valid for 15 minutes.
```

**Copy the address of the page that fails, not the URL you were given.** Anthropic's authorization URL
contains a literal `code=true` parameter, so pasting it back would otherwise look like a valid code and be
rejected by the provider as `invalid_grant`. That mistake is now caught and named.

### Why the login has two halves

The PKCE verifier is generated with the URL and must be sent with the code. Holding it only in memory means
an interrupted login — Ctrl+C, a closed pipe, a walk to another machine — destroys it, and any code from
that URL can never be exchanged; the provider reports only `PKCE verification failed`.

`begin_login` therefore writes the verifier, state and redirect URI to `pending.json` (`0600`, valid 15
minutes) and `complete_login` reads it back, so the two halves can run in separate invocations. An
interrupted login is resumed with `--code`, and a code from a superseded attempt is reported as such rather
than as a PKCE failure.

### Using a browser on this host instead

Forward the port when you connect and pass `--browser`:

```sh
ssh -L 53692:localhost:53692 <host>     # Anthropic
ssh -L 56121:localhost:56121 <host>     # xAI
litellm anthropic-oauth login --browser
```

ChatGPT needs none of this: its device-code flow prints a code you enter on any device, so it works over
SSH unchanged.

**Log in before starting the proxy.** Left to itself, a provider logs in lazily on the first request that
needs a token — and the ChatGPT device flow prints a code to stdout and blocks for up to fifteen minutes. On
a server that is a stalled request and an invisible prompt. These commands exist so the login is a deliberate
step with a terminal attached.

## Configuring a model

Credentials are stored per provider, so a deployment only has to opt in. An explicit `api_key` always wins,
which is what makes it easy to keep a subscription-backed and a key-backed deployment of the same model side
by side and route between them.

```yaml
model_list:
  - model_name: claude-sub                 # billed to the Claude Pro/Max subscription
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      use_anthropic_oauth: true

  - model_name: grok-sub                   # billed to the SuperGrok subscription
    litellm_params:
      model: xai/grok-build-0.1
      use_xai_oauth: true

  - model_name: codex-sub                  # billed to the ChatGPT subscription
    litellm_params:
      model: chatgpt/gpt-5

  - model_name: claude-key                 # the same model on an API key
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
```

The ChatGPT provider is subscription-only and needs no flag; it reaches `chatgpt.com/backend-api/codex`
rather than the public API.

**An API key in the environment wins over the flag.** `use_xai_oauth` and `use_anthropic_oauth` are only
consulted once no key has been resolved, and resolution includes `XAI_API_KEY` / `ANTHROPIC_API_KEY` from the
environment. A deployment carrying the flag will therefore bill to the key, silently, wherever one is set —
which looks exactly like the subscription working. To check which was used, unset the environment variable
and confirm the deployment still answers, or watch for the provider's OAuth base URL in a debug log.

## Where credentials are stored

By default, on disk:

| provider | path | override |
|---|---|---|
| Anthropic | `~/.config/litellm/anthropic_oauth/auth.json` | `ANTHROPIC_OAUTH_TOKEN_DIR` |
| xAI | `~/.config/litellm/xai_oauth/auth.json` | `XAI_OAUTH_TOKEN_DIR` |
| ChatGPT | `~/.config/litellm/chatgpt/auth.json` | `CHATGPT_TOKEN_DIR` |

Each file holds a refresh token, which is a long-lived credential: the directory is `0700`, the file `0600`,
and writes go to a temporary file renamed over the target so a reader never sees a half-written record.
`status` never prints the tokens themselves.

### In the database

A file on one host is wrong for a gateway: it does not survive a container without a mounted volume, and
replicas each have their own, so each has to log in separately and each refreshes independently.

Where the proxy has a `DATABASE_URL`, credentials are read from and written to it instead — no configuration
needed. Set `general_settings.store_oauth_credentials_in_db: false` to keep using files.

Move an existing login in, then remove the local copy:

```sh
litellm anthropic-oauth push      # local file -> database
litellm xai-oauth push
rm ~/.config/litellm/anthropic_oauth/auth.json

litellm anthropic-oauth pull      # database -> local file, to debug or take away
```

Records live in `LiteLLM_CredentialsTable` — the table the proxy already uses for provider credentials — as
`oauth/<provider>`, so there is no new table and no migration. **Every field is encrypted** with LiteLLM's
own helper, so tokens at rest are protected by `LITELLM_SALT_KEY` (or the master key) like any other stored
credential; a row read directly from Postgres contains only ciphertext. Set `LITELLM_SALT_KEY` before
pushing, and keep it: without it the records decrypt to nothing and every provider reads as logged out.

Refresh tokens rotate, so two replicas refreshing from one record would each invalidate the other's. A
refresh therefore takes a Postgres advisory lock for the length of its read-refresh-write, keyed per
provider so unrelated providers do not queue behind each other, and released by the transaction ending —
including if the process dies. Reads are cached for a few seconds so a burst of requests is one query.

If the database is unreachable, a read falls back to the local file where one exists, so a blip degrades
rather than taking every model offline.

## How each provider works

| provider | flow | token use |
|---|---|---|
| Anthropic | authorization code + PKCE, loopback `127.0.0.1:53692`, token endpoint takes JSON and echoes `state` | `sk-ant-oat…` becomes a Bearer header plus `anthropic-beta: oauth-2025-04-20`, and `x-api-key` is dropped |
| xAI | authorization code + PKCE, OIDC discovery against `auth.x.ai`, loopback `127.0.0.1:56121` | Bearer, against the OAuth-specific API base |
| ChatGPT | device code against `auth.openai.com`, polled until the browser confirms | Bearer plus `ChatGPT-Account-Id` (read from the JWT) and Codex's `originator` headers |

Refresh happens inside `get_access_token()` on the request path, under a lock so concurrent requests perform
one refresh rather than several, with a skew so a token is replaced before it expires mid-request rather than
after a request has already failed.

`llms/base_llm/oauth.py` holds what these share: PKCE, the loopback callback server, the credential file,
remote-session detection and the pasted-code parser. `llms/xai/oauth.py` predates it and still keeps its own
callback server and credential file, but now uses the shared parser; it can adopt the rest unchanged.

## Interaction with the session manager

The ChatGPT provider forwards `litellm_session_id` upstream as a `session_id` header, which is the same
identity the session manager keys a session's `prompt_cache_key` on — so a compacted session and the Codex
backend agree on what the session is without any extra wiring.

## Anthropic: why the login alone is not enough

A Claude Pro/Max OAuth token is accepted by Anthropic's account endpoints but refused for inference on
`/v1/messages`, which answers a generic `rate_limit_error` with quota to spare. Measured here: the same
token returned 200 from `api.anthropic.com/api/oauth/usage` (five-hour utilisation 77%, seven-day 51%)
while every `/v1/messages` call was rejected. Anthropic issues these tokens to Claude Code and appears to
serve inference only to it.

So a Claude subscription is spent by **running Claude Code**, which is what openclaw does and what the
`claude_cli` provider does here. It uses Claude Code's documented headless mode (`claude --print`); it does
not imitate Claude Code to the API.

```yaml
litellm_settings:
  custom_provider_map:
    - provider: claude_cli
      custom_handler: litellm.llms.claude_cli.handler.claude_cli_handler

model_list:
  - model_name: claude-sub
    litellm_params:
      model: claude_cli/claude-sonnet-5-5
```

It registers through `custom_provider_map`, so no core dispatch code changes and the fork stays easy to
rebase. `litellm anthropic-oauth login` supplies the credential; `ANTHROPIC_API_KEY` or an explicit
`api_key` is used instead when present.

| setting | default | meaning |
|---|---|---|
| `claude_cli_allowed_tools` | none | tools to permit, e.g. `["Read", "Grep"]` |
| `claude_cli_timeout` | 600 | seconds before the subprocess is killed |
| `claude_cli_config_dir` | Claude Code's own | `CLAUDE_CONFIG_DIR` for the subprocess |
| `CLAUDE_CLI_PATH` (env) | `claude` | where the binary lives |

**Claude Code's built-in host tools are off unless a deployment names them.** Claude Code can read files and run shell commands, so a
gateway serving requests with tools enabled would hand every caller code execution on the proxy host. Enable
them only where every caller is trusted

### Client-executed tools through the MCP sidecar

Requests containing function tool schemas use a different path. Claude Code runs with built-in tools
disabled and a generated, strict MCP configuration pointing to `llms/claude_cli/mcp_sidecar.py`.
The sidecar forwards selected calls through a private Unix socket to the gateway's relay in
`llms/claude_cli/tool_bridge.py`. The gateway returns structured function calls to the client, which
executes them and returns correlated results. These function schemas never enable host file or shell tools

The Claude subprocess remains alive while those tool rounds are pending. Completed turns reconstruct
from the client's history on the next request, so edited or branched history does not accidentally
continue an old subprocess context. Function tools require a client session ID and an async completion
or streaming call. Only absent/default or `auto` tool choice is supported, and image input is rejected

The gateway relay and the MCP sidecar are parts of the same design, not alternative execution modes.
See [session-store operations](../sessions/OPERATIONS.md) for restart recovery, session ownership,
steering input, worker affinity and verification

The credential goes to the subprocess on a pipe, whose descriptor number is passed in
`CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR` — never the token itself in its environment or argv, both of which
are readable through `/proc`. `ANTHROPIC_API_KEY` and `CLAUDE_CODE_OAUTH_TOKEN` are scrubbed from the child
environment so an ambient one cannot be used by accident.

### How a conversation continues

The text-only path without client function schemas uses the transcript behavior described here.
Client-executed tools instead use the sidecar lifecycle above

Claude Code owns the transcript. The first turn opens a session with `--session-id`; later turns pass
`--resume` and send only the newest message, so Claude Code keeps its own prefix warm — a second turn costs
a handful of input tokens rather than the whole history. The session id is a UUID derived from the owner and
`litellm_session_id`, so nothing is stored to map between them and a gateway restart resumes the same
conversation. Two API keys cannot collide: the owner is part of the derivation.

If Claude Code no longer has the session, the turn is retried once as a new session carrying the whole
conversation, so a cleared history costs a cache miss rather than an error.

Claude Code's own system prompt is **replaced** rather than appended to. Left in place it costs about 3,700
tokens per conversation and, with tools disabled, makes the model emit tool-call syntax as plain text
(`<invoke name="Write">…`) because the prompt still advertises tools.

Two consequences worth knowing:

- **The session manager does not drive this provider.** Claude Code owns the transcript, its caching and its
  own compaction. Layering the gateway's transcript on top would be two owners for one conversation, which
  is the duplication this fork exists to remove. The gateway still keeps a *record* of it; see below.
- **A subprocess per request** costs process startup: the first token arrives after roughly six seconds.
  Replies stream (`--output-format stream-json --include-partial-messages`), so everything after that
  arrives as it is written. Each run is its own process group, and a caller who disconnects mid-reply ends it
  and anything it started.
- **No login of its own is used.** `claude_cli_config_dir` gives Claude Code a home apart from the interactive
  one, and it authenticates only with the token on the pipe. Verified: from an empty home the gateway's token
  answers and a bogus one fails.

### Where the transcript goes

Because only the newest message passes through the proxy, a stored transcript built from what the gateway
saw would hold a conversation's last exchange and nothing else -- no history, and nothing Claude Code did to
produce the answer. So the record is read from where it actually exists.

After each turn the provider reports the session; `litellm/proxy/claude_cli_sync.py` reads Claude Code's own
transcript (`~/.claude/projects/<project>/<session-id>.jsonl`) and imports it through the same path as any
other harness history, as `source = claude_code`. Re-importing appends only what is new, so a turn costs one
read of a file the harness has just written. The session row carries the caller's `litellm_session_id`, so
the stored transcript lines up with their own session identity, and the owner is the key that was billed.

This is on wherever the session manager already stores transcripts (`session_manager.store_transcripts` with
a database), and off everywhere else -- a library call or a CLI login stores nothing. It runs after the
answer has gone back, and a failure to store is logged, never raised: a gap in the record is not a failed
request.

The same reader backs `python -m litellm.proxy.session_import claude-code`, which archives sessions Claude
Code ran outside the proxy entirely.

## What this deliberately does not do

- **No impersonation.** Getting a Pro/Max token accepted on `/v1/messages` would mean presenting the
  gateway as Claude Code. That is circumventing an access control Anthropic put there deliberately, so the
  provider runs the real client instead.
- **No credential pool.** Hermes keeps several credentials per provider and fails over between them on rate
  limits. LiteLLM's router already does multi-deployment failover, so a pool here would be a second
  mechanism for the same thing.
- **No third-party login dependency.** The published options either pull an entire agent framework into the
  gateway (hermes-agent, with a pinned `openai` that fights LiteLLM's own) or are lightly-maintained ports;
  a dependency in the auth path is the worst place to accept that risk.
