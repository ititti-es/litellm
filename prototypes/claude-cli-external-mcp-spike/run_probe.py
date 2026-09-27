#!/usr/bin/env python3
"""Run offline checks, or one guarded Claude CLI MCP feasibility probe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from external_executor import EXPECTED_REQUEST, SYNTHETIC_RESULT, SyntheticExecutor, executor_url
from mcp_forwarder import request_external_result

ROOT = Path(__file__).resolve().parent
T = TypeVar("T")


def run_with_executor(callback: Callable[[str, SyntheticExecutor], T]) -> T:
    server = SyntheticExecutor()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return callback(executor_url(server), server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_offline() -> None:
    def check(url: str, server: SyntheticExecutor) -> None:
        assert request_external_result(url) == SYNTHETIC_RESULT
        assert server.received_requests == [EXPECTED_REQUEST]

    run_with_executor(check)
    sys.stdout.write("OFFLINE_PASS: forwarder waited for one separate executor result; no execution occurred internally.\n")


def run_claude() -> None:
    def check(url: str, server: SyntheticExecutor) -> None:
        config = {
            "mcpServers": {
                "external-executor": {
                    "command": sys.executable,
                    "args": [str(ROOT / "mcp_forwarder.py")],
                    "env": {"SYNTHETIC_EXECUTOR_URL": url},
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            command = [
                "claude",
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--restricted",
                "--tools",
                "",
                "--strict-mcp-config",
                "--mcp-config",
                config_file.name,
                "--allowed-tools",
                "mcp__external-executor__request_external_executor",
                "--permission-mode",
                "dontAsk",
                "--permission-prompts",
                "none",
                "--max-budget-usd",
                "0.01",
                "Use mcp__external-executor__request_external_executor exactly once. "
                "After its result, reply with exactly CONTINUED_AFTER_EXTERNAL_RESULT.",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )

        if completed.returncode != 0:
            diagnostics = (completed.stderr or completed.stdout)[-500:]
            raise RuntimeError(f"Claude CLI failed with exit code {completed.returncode}: {diagnostics}")
        if server.received_requests != [EXPECTED_REQUEST]:
            raise RuntimeError(f"expected one executor request, got {len(server.received_requests)}")
        if SYNTHETIC_RESULT not in completed.stdout:
            raise RuntimeError("external executor result was not surfaced in Claude stream output")
        if "CONTINUED_AFTER_EXTERNAL_RESULT" not in completed.stdout:
            raise RuntimeError("Claude did not continue after the external result")

    run_with_executor(check)
    sys.stdout.write("CLAUDE_PASS: one MCP request paused at the external executor and Claude continued after its result.\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--claude", action="store_true")
    arguments = parser.parse_args()
    if arguments.offline:
        run_offline()
    else:
        run_claude()


if __name__ == "__main__":
    main()
