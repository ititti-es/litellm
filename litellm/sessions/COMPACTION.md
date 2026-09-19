# Warm-cache compaction policies

The session manager compacts a session just before its upstream prompt cache is predicted to expire. It sends the
provider the exact prefix the next turn would send, plus one instruction message, so the long history is read at
cached rates. The session is then rebased onto the summary and its latest exchange(s).

A **policy** decides what that instruction says and how the summary is put back. Policies are data: stored in
`LiteLLM_CompactionPolicy`, editable without code.

## Built-in presets

| name | emulates | summary is replayed as |
|---|---|---|
| `default` | — | a user message: "Summary of the conversation so far …" |
| `opencode` | opencode's anchored summary (Objective / Important Details / Work State / Next Move / Relevant Files) | user "What did we do so far?" then an assistant message with the summary |
| `openclaw` | openclaw's context checkpoint (Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context) | a user message wrapped in openclaw's `<summary>` framing |
| `hermes` | hermes-agent's checkpoint (Historical Task Snapshot, Completed Actions, Errors & Fixes, …, credential redaction) | a user message behind hermes's "[CONTEXT COMPACTION — REFERENCE ONLY]" handoff |
| `off` | — | never compacts |

Each has a first-compaction prompt and an update prompt used when the history already starts from a summary. The
sources and the one adaptation made (the conversation is the cached prefix, not a pasted block) are recorded in
`compaction_presets.py`.

## When a session is compacted at all

Timing comes from the cache clock: just before the predicted expiry. Whether it happens is a size decision, because
a compaction trades verbatim history for a summary. A modest session is cheap to resend cold after a break and its
detail is worth more than the tokens saved, so it is left alone. A session is compacted only when its next prefix
clears every bar:

| setting | default | meaning |
|---|---|---|
| `min_prefix_tokens` | 20000 | absolute floor |
| `min_context_fraction` | 0.5 | share of the model's context window; an unknown window never clears it |
| `enabled` | true | off switch |

With the defaults, a 20k-token session on a 256k-window model is never compacted; one at 128k is.

The bars are layered, each field falling back independently:

```yaml
general_settings:
  session_manager:
    compaction:
      enabled: true
      min_prefix_tokens: 20000          # proxy-wide
      min_context_fraction: 0.5
      models:
        grok:                           # model_list name (or the provider model string)
          min_context_fraction: 0.7
        some-unknown-model:
          context_window: 131072        # needed when LiteLLM does not know the window
        cheap-model:
          enabled: false
```

and a policy may set `enabled`, `min_prefix_tokens` or `min_context_fraction` in its settings, which win over the
proxy's for sessions using it. The built-in `off` policy sets `enabled: false`, so a client can opt a session out
with `metadata: {"compaction_policy": "off"}`.

## Choosing a policy

1. Per request: `metadata: {"compaction_policy": "opencode"}` on the Responses call (HTTP or WebSocket). A harness
   sets this to get its own behaviour back; the latest turn's choice applies.
2. Proxy default: `general_settings.session_manager.compaction.policy: opencode`.
3. Otherwise `default`.

A name resolves to a stored policy first, then a built-in preset. An unknown or invalid name falls through to the
next choice, and is logged.

## Customizing

```sh
export DATABASE_URL=postgresql://…
python -m litellm.proxy.compaction_policies list
python -m litellm.proxy.compaction_policies seed opencode          # store an editable copy of a preset
python -m litellm.proxy.compaction_policies show opencode > p.yaml
$EDITOR p.yaml                                                      # prompts are plain YAML text blocks
python -m litellm.proxy.compaction_policies set p.yaml              # takes effect on the next compaction
python -m litellm.proxy.compaction_policies delete opencode         # back to the built-in preset
```

A new policy is just a new `name`:

```yaml
name: terse
source: custom
instructions: |
  The conversation above is about to be compacted. In at most ten bullets, record the goal, decisions,
  exact identifiers and values, and the next step. Today is {{date}}. Do not call tools.
settings:
  update_instructions: |
    The conversation above starts from an earlier summary. Merge it with what happened since, in at most
    ten bullets; the newer messages win where they conflict.
  summary_role: user            # or assistant
  summary_lead_in: null         # a user message before an assistant summary
  summary_prefix: "Notes so far:\n\n"
  summary_suffix: ""
  keep_turns: 1                 # latest user turns kept verbatim
  max_output_tokens: 2048       # null for no cap
  min_context_fraction: 0.6     # optional: this policy's own bar (also enabled, min_prefix_tokens)
```

`{{date}}` is replaced with the current UTC date. What a policy cannot change is the prefix itself: model,
instructions, tools and cache key always match the session's last turn, since changing them would miss the cache.

Each compaction records the policy name and source in the summary item's meta, alongside what it covered and what
it cost.
