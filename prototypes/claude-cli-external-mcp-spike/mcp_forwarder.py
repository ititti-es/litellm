#!/usr/bin/env python3
"""MCP server that relays one fixed request to a separate synthetic executor."""

from __future__ import annotations

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP

REQUEST = {"operation": "synthetic_probe", "payload": "mcp-pause-test"}
TIMEOUT_SECONDS = 10
mcp = FastMCP("external-executor")


def request_external_result(executor_url: str) -> str:
    """Relay the fixed synthetic request and wait for the external result."""
    if not executor_url.startswith("http://127.0.0.1:"):
        raise ValueError("spike executor must be loopback HTTP")

    request = Request(
        executor_url,
        data=json.dumps(REQUEST).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            result = json.loads(response.read())
    except (OSError, URLError, json.JSONDecodeError) as error:
        raise RuntimeError("synthetic executor did not return a valid result") from error

    if result != {"result": "EXTERNAL_EXECUTOR_RESULT: synthetic-ok"}:
        raise RuntimeError("synthetic executor returned an unexpected result")
    return result["result"]


@mcp.tool()
def request_external_executor() -> str:
    """Request one fixed synthetic result from the separate external executor."""
    return request_external_result(os.environ["SYNTHETIC_EXECUTOR_URL"])


if __name__ == "__main__":
    mcp.run(transport="stdio")
