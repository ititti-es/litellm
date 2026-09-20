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
```

Flags on `login`:

| flag | effect |
|---|---|
| `--force` | ignore a stored token and log in again |
| `--no-browser` | print the authorization URL instead of opening a browser |
| `--manual` | print the URL and prompt for the code, for a host whose browser cannot reach its loopback port (Anthropic only) |

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

## Where credentials are stored

| provider | path | override |
|---|---|---|
| Anthropic | `~/.config/litellm/anthropic_oauth/auth.json` | `ANTHROPIC_OAUTH_TOKEN_DIR` |
| xAI | `~/.config/litellm/xai_oauth/auth.json` | `XAI_OAUTH_TOKEN_DIR` |
| ChatGPT | `~/.config/litellm/chatgpt/auth.json` | `CHATGPT_TOKEN_DIR` |

Each file holds a refresh token, which is a long-lived credential: the directory is `0700`, the file `0600`,
and writes go to a temporary file that is renamed over the target so a reader never sees a half-written
record. `status` never prints the tokens themselves.

These are files on one host, so they do not survive a container without a mounted volume and are not shared
between replicas. Each replica must log in, or share the directory.

## How each provider works

| provider | flow | token use |
|---|---|---|
| Anthropic | authorization code + PKCE, loopback `127.0.0.1:53692`, token endpoint takes JSON and echoes `state` | `sk-ant-oat…` becomes a Bearer header plus `anthropic-beta: oauth-2025-04-20`, and `x-api-key` is dropped |
| xAI | authorization code + PKCE, OIDC discovery against `auth.x.ai`, loopback `127.0.0.1:56121` | Bearer, against the OAuth-specific API base |
| ChatGPT | device code against `auth.openai.com`, polled until the browser confirms | Bearer plus `ChatGPT-Account-Id` (read from the JWT) and Codex's `originator` headers |

Refresh happens inside `get_access_token()` on the request path, under a lock so concurrent requests perform
one refresh rather than several, with a skew so a token is replaced before it expires mid-request rather than
after a request has already failed.

`llms/base_llm/oauth.py` holds what these share: PKCE, the loopback callback server, and the credential file.
`llms/xai/oauth.py` predates it and keeps its own copies; it can adopt the shared helpers unchanged.

## Interaction with the session manager

The ChatGPT provider forwards `litellm_session_id` upstream as a `session_id` header, which is the same
identity the session manager keys a session's `prompt_cache_key` on — so a compacted session and the Codex
backend agree on what the session is without any extra wiring.

## What this deliberately does not do

- **No credential pool.** Hermes keeps several credentials per provider and fails over between them on rate
  limits. LiteLLM's router already does multi-deployment failover, so a pool here would be a second
  mechanism for the same thing.
- **No third-party login dependency.** The published options either pull an entire agent framework into the
  gateway (hermes-agent, with a pinned `openai` that fights LiteLLM's own) or are lightly-maintained ports;
  a dependency in the auth path is the worst place to accept that risk.
