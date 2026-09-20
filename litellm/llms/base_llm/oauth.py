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
import sys
import time
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final, Protocol, TypedDict
from urllib.parse import parse_qs, urlparse

from typing_extensions import ReadOnly

from litellm._logging import verbose_logger

LOOPBACK_HOST: Final = "127.0.0.1"
CALLBACK_POLL_SECONDS: Final = 1

# Query parameters that belong to an authorization *request*; seeing one means the user pasted the URL they
# were asked to open rather than where it sent them.
_AUTHORIZE_URL_MARKERS: Final = ("code_challenge", "response_type", "client_id", "scope")


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


def is_remote_session() -> bool:
    """Whether this process has no browser it could usefully open.

    Over SSH a loopback redirect would arrive at the *client's* machine, where nothing is listening, and
    `webbrowser.open` can report success having done nothing. Callers use this to print the URL instead.
    """
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY") or os.environ.get("SSH_CLIENT"):
        return True
    return sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")


def parse_authorization_input(value: str) -> tuple[str, str | None]:
    """Parse what a user pastes back when no browser can reach the loopback redirect.

    Providers show the code as `<code>#<state>`, and a user may equally paste the whole redirect URL out of
    the browser's address bar, so both are accepted. Raises ValueError for anything else.
    """
    text: Final = value.strip()
    if not text:
        raise ValueError("No authorization code provided")
    if text.startswith("http://") or text.startswith("https://"):
        params: Final = parse_qs(urlparse(text).query)
        # An authorization URL is easy to paste back by mistake, and Anthropic's carries a literal
        # `code=true`, which would otherwise be read as the code itself and rejected as invalid by the
        # provider. These parameters only ever appear on the request, never on the redirect back.
        if any(marker in params for marker in _AUTHORIZE_URL_MARKERS):
            raise ValueError(
                "That is the authorization URL, not the result of opening it. Open it in a browser, approve "
                "the request, and copy the address of the 127.0.0.1 page it then fails to load."
            )
        code: Final = _first(params, "code")
        if not code:
            raise ValueError("Redirect URL contained no authorization code")
        return code.split("#")[0], _first(params, "state")
    if "#" in text:
        raw_code, _, raw_state = text.partition("#")
        return raw_code, raw_state or None
    return text, None


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
    redirect_host: str = LOOPBACK_HOST,
    allow_ephemeral: bool = True,
) -> tuple[CallbackServer, str]:
    """Bind a loopback callback server, returning it and the redirect URI to advertise.

    The socket always binds the numeric loopback address, but `redirect_host` is what goes into the
    redirect URI, because a provider matches that string against what the client registered and some
    register `localhost` rather than `127.0.0.1`.

    `allow_ephemeral` falls back to any free port when the preferred one is busy. That only helps where the
    provider accepts an arbitrary port; where it does not, failing is better than sending a redirect URI
    that is certain to be rejected.
    """
    last_error: OSError | None = None
    for port in (preferred_port, 0) if allow_ephemeral else (preferred_port,):
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
        return server, f"http://{redirect_host}:{actual_port}{callback_path}"
    raise OSError(f"Could not start {brand} OAuth callback server on port {preferred_port}: {last_error}")


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


PENDING_LOGIN_TTL_SECONDS: Final = 900


class PendingLogin(TypedDict):
    """A login that has been started but not yet exchanged for a token.

    The PKCE verifier exists only in the process that built the authorize URL, so a single-shot command
    throws it away if the user cannot answer the prompt right then -- and any code from that URL becomes
    unusable. Persisting it lets the two halves happen in separate invocations, which is what a
    non-interactive shell or a slow trip to another machine's browser requires.
    """

    verifier: ReadOnly[str]
    state: ReadOnly[str]
    redirect_uri: ReadOnly[str]
    authorize_url: ReadOnly[str]
    token_endpoint: ReadOnly[str]
    created_at: ReadOnly[float]


def pending_is_fresh(pending: PendingLogin, now: float | None = None) -> bool:
    created: Final = pending.get("created_at")
    if not isinstance(created, (int, float)):
        return False
    return (now if now is not None else time.time()) - float(created) < PENDING_LOGIN_TTL_SECONDS


class CredentialStore(Protocol):
    """Where a provider's OAuth record lives.

    A file is the default and is right for a workstation. A gateway wants the record somewhere its replicas
    share and a container restart does not lose, which is what the proxy's database-backed store provides.
    """

    def read(self) -> object | None: ...

    def write(self, data: object) -> None: ...

    def delete(self) -> bool: ...

    def guard(self) -> AbstractContextManager[None]:
        """Hold exclusive access across the read-refresh-write of a token.

        Refresh tokens rotate: two processes refreshing from the same record would each invalidate the
        other's. In-process locking is enough for a single host; a shared store has to reach further.
        """
        ...


@contextmanager
def _no_guard() -> Generator[None]:
    yield


_STORE_FACTORY: list[Callable[[str, str], CredentialStore]] = []  # mutable-ok: single-slot registry


def set_credential_store_factory(factory: Callable[[str, str], CredentialStore] | None) -> None:
    """Choose where provider credentials are read and written.

    The proxy calls this at startup when it has a database, so the authenticators keep their file default
    everywhere else -- a CLI login on a workstation should not need a database.
    """
    _STORE_FACTORY.clear()
    if factory is not None:
        _STORE_FACTORY.append(factory)


def credential_store_for(provider: str, path: str) -> CredentialStore:
    """Return the store a provider should use: whatever was registered, else a file at `path`."""
    if _STORE_FACTORY:
        try:
            return _STORE_FACTORY[0](provider, path)
        except Exception as exc:  # noqa: BLE001  # any store failure must still leave a usable file login
            verbose_logger.warning("Falling back to file credentials for %s: %s", provider, exc)
    return TokenFile(path)


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

    def guard(self) -> AbstractContextManager[None]:
        """No-op: one host's processes are serialised by the authenticator's own lock."""
        return _no_guard()

    def _ensure_dir(self) -> None:
        os.makedirs(self.directory, mode=0o700, exist_ok=True)
        self._chmod(self.directory, 0o700)

    def _chmod(self, path: str, mode: int) -> None:
        try:
            os.chmod(path, mode)
        except OSError:
            verbose_logger.debug("Could not chmod OAuth credential path %s", path)
