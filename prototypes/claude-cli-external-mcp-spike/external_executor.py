#!/usr/bin/env python3
"""A fixed-result executor used only by the external-MCP spike."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

EXPECTED_REQUEST = {"operation": "synthetic_probe", "payload": "mcp-pause-test"}
SYNTHETIC_RESULT = "EXTERNAL_EXECUTOR_RESULT: synthetic-ok"


class SyntheticExecutor(ThreadingHTTPServer):
    """Loopback-only server that records requests but never executes anything."""

    received_requests: list[dict[str, Any]]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ExecutorHandler)
        self.received_requests = []


class ExecutorHandler(BaseHTTPRequestHandler):
    server: SyntheticExecutor

    def do_POST(self) -> None:
        if self.path != "/execute":
            self.send_error(404)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(content_length))
        except (ValueError, json.JSONDecodeError):
            self.send_error(400, "invalid JSON")
            return

        if request != EXPECTED_REQUEST:
            self.send_error(422, "unexpected synthetic request")
            return

        self.server.received_requests.append(request)
        body = json.dumps({"result": SYNTHETIC_RESULT}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        """Keep the probe output free of HTTP request details."""


def executor_url(server: SyntheticExecutor) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/execute"
