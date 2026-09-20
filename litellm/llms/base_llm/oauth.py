"""Shared primitives for provider OAuth logins.

A subscription login (Claude Pro/Max, SuperGrok, ChatGPT Plus) is the same shape everywhere: PKCE, an
authorization code delivered to a loopback redirect, and a refresh token kept in a file only the user can
read. This module holds those parts so each provider module carries only what is actually provider-specific:
its endpoints, client id, scopes and token record.

`litellm/llms/xai/oauth.py` predates this module and carries its own copies of the callback server and token
file. It can adopt these helpers without any change in behaviour.
"""

import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final, TypedDict
from urllib.parse import parse_qs, urlparse

from typing_extensions import ReadOnly

from litellm._logging import verbose_logger

LOOPBACK_HOST: Final = "127.0.0.1"
CALLBACK_POLL_SECONDS: Final = 1


class CallbackResult(TypedDict):
    """The query parameters an authorization server sends back to the loopback redirect."""

    code: ReadOnly[str | None]
    state: ReadOnly[str | None]
    error: ReadOnly[str | None]
    error_description: ReadOnly[str | None]


def pkce_pair() -> tuple[str, str]:
    """Return a (verifier, challenge) pair for PKCE S256."""
    verifier: Final = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge: Final = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def new_state() -> str:
    """Return an unguessable value binding an authorization request to its callback."""
    return uuid.uuid4().hex


def _first(params: Mapping[str, Sequence[str]], key: str) -> str | None:
    values: Final = params.get(key)
    return values[0] if values else None


def _page(heading: str) -> bytes:
    return f"<html><body><h1>{heading}</h1>You can close this tab.</body></html>".encode()


class CallbackServer(HTTPServer):
    """A loopback HTTP server that accepts exactly one OAuth redirect."""

    expected_state: str
    callback_path: str
    brand: str
    callback_result: CallbackResult | None


class _CallbackHandler(BaseHTTPRequestHandler):
    # The stdlib stubs type `server` as BaseServer; CallbackServer is the only server this handler is
    # registered on, and it is what supplies expected_state / callback_path / brand.
    server: CallbackServer  # pyright: ignore[reportIncompatibleVariableOverride]  # see comment above

    def do_GET(self) -> None:
        parsed: Final = urlparse(self.path)
        if parsed.path != self.server.callback_path:
            self.send_response(404)
            self.end_headers()
            return

        params: Final = parse_qs(parsed.query)
        result: Final[CallbackResult] = {  # mutable-ok: the parsed callback, built once and never mutated
            "code": _first(params, "code"),
            "state": _first(params, "state"),
            "error": _first(params, "error"),
            "error_description": _first(params, "error_description"),
        }
        self.server.callback_result = result

        if result["state"] != self.server.expected_state:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_page(f"{self.server.brand} authorization state mismatch."))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        heading: Final = (
            f"{self.server.brand} authorization failed."
            if result["error"]
            else f"{self.server.brand} authorization received."
        )
        self.wfile.write(_page(heading))

    def log_message(self, format: str, *args: object) -> None:
        return


def start_callback_server(
    expected_state: str,
    preferred_port: int,
    callback_path: str,
    brand: str,
) -> tuple[CallbackServer, str]:
    """Bind a loopback callback server, returning it and the redirect URI to register.

    The preferred port is tried first because some authorization servers only allow redirect URIs they
    already know. An ephemeral port is the fallback for when that one is busy.
    """
    last_error: OSError | None = None
    for port in (preferred_port, 0):
        try:
            server = CallbackServer((LOOPBACK_HOST, port), _CallbackHandler)
        except OSError as exc:
            last_error = exc
            continue
        server.expected_state = expected_state
        server.callback_path = callback_path
        server.brand = brand
        server.callback_result = None
        actual_port = server.server_address[1]  # rebind-ok: one binding per loop iteration
        return server, f"http://{LOOPBACK_HOST}:{actual_port}{callback_path}"
    raise OSError(f"Could not start {brand} OAuth callback server: {last_error}")


def wait_for_callback(server: CallbackServer, timeout_seconds: float) -> CallbackResult | None:
    """Serve requests until the redirect arrives, or None if it does not within the timeout."""
    server.timeout = CALLBACK_POLL_SECONDS
    deadline: Final = time.time() + timeout_seconds
    try:
        while time.time() < deadline:
            server.handle_request()
            if server.callback_result is not None:
                return server.callback_result
    finally:
        server.server_close()
    return None


class TokenFile:
    """A JSON credential file readable only by its owner.

    Writes go to a temporary file in the same directory and are renamed over the target, so a crash or a
    concurrent reader never observes a half-written record.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.directory = os.path.dirname(path)

    def read(self) -> object | None:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def write(self, data: object) -> None:
        self._ensure_dir()
        tmp_file: Final = os.path.join(self.directory, f".{os.path.basename(self.path)}.{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL  # rebind-ok: extended below when the platform allows
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd: Final = os.open(tmp_file, flags, 0o600)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.path)
            self._chmod(self.path, 0o600)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp_file)
            except OSError:
                pass
            raise

    def delete(self) -> bool:
        try:
            os.unlink(self.path)
            return True
        except OSError:
            return False

    def _ensure_dir(self) -> None:
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self._chmod(self.directory, 0o700)

    def _chmod(self, path: str, mode: int) -> None:
        try:
            os.chmod(path, mode)
        except OSError:
            verbose_logger.debug("Could not chmod OAuth credential path %s", path)
