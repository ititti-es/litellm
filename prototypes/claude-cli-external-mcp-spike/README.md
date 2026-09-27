# Claude CLI external MCP feasibility spike

This isolated spike checks whether Claude Code can make one synthetic MCP tool call, wait while a separate loopback executor returns a result, then continue its response

It does not modify LiteLLM routes or handlers. The executor accepts only `synthetic_probe` and returns a fixed result. It runs no commands, reads no files, and accepts no credentials

## Run

```bash
uv run --no-sync python prototypes/claude-cli-external-mcp-spike/run_probe.py --offline
```

The offline mode validates the forwarding boundary without contacting Claude

```bash
uv run --no-sync python prototypes/claude-cli-external-mcp-spike/run_probe.py --claude
```

Claude mode makes one short subscription request. It uses `--restricted`, disables built-in tools, loads only a generated local MCP config, allows only this tool, and sets `--max-budget-usd 0.01`. The cap applies to API-key billing and is retained as a guard for a subscription login

## Architecture

```
Claude CLI -> stdio MCP forwarder -> loopback synthetic executor -> fixed result
```

The MCP forwarder blocks until the executor returns. It can only validate and relay the fixed request. A real bridge needs an authenticated job and result protocol, request IDs, cancellation, timeouts, and result validation

## Pass Criteria

Claude mode passes only if exactly one executor request arrives with `synthetic_probe`, the stream contains `EXTERNAL_EXECUTOR_RESULT: synthetic-ok`, and Claude returns `CONTINUED_AFTER_EXTERNAL_RESULT`

This CLI path is synchronous. It can wait for the executor while the CLI process and MCP server stay open. It cannot return a durable pending tool call and resume it later without an open session
