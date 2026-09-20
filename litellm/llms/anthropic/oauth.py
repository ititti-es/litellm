"""Anthropic OAuth login for Claude Pro/Max subscriptions.

LiteLLM already *consumes* an Anthropic OAuth token: `optionally_handle_anthropic_oauth` in `common_utils.py`
recognises the `sk-ant-oat` prefix and swaps `x-api-key` for a Bearer header plus the OAuth beta header. What
was missing was a way to obtain and refresh one. This module fills that gap, mirroring the shape of
`litellm/llms/xai/oauth.py` so the two read the same way.

The flow is an authorization code with PKCE against `claude.ai`, redirected to a loopback port. Anthropic's
token endpoint takes JSON (not form encoding) and echoes `state` back in the exchange, which is why this does
not simply reuse the xAI request bodies.
"""

import json
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, TypedDict
from urllib.parse import urlencode

import httpx
from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly

from litellm._logging import verbose_logger
from litellm.llms.base_llm.oauth import (
    PendingLogin,
    TokenFile,
    new_state,
    pending_is_fresh,
    pkce_pair,
    start_callback_server,
    wait_for_callback,
)
from litellm.llms.base_llm.oauth import parse_authorization_input as _parse_authorization_input
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client
from litellm.secret_managers.main import get_secret_str

ANTHROPIC_OAUTH_CLIENT_ID: Final = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_OAUTH_AUTHORIZE_URL: Final = "https://claude.ai/oauth/authorize"
ANTHROPIC_OAUTH_TOKEN_URL: Final = "https://platform.claude.com/v1/oauth/token"
ANTHROPIC_OAUTH_SCOPE: Final = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload"
)
ANTHROPIC_OAUTH_REDIRECT_PORT: Final = 53692
ANTHROPIC_OAUTH_REDIRECT_PATH: Final = "/callback"
ANTHROPIC_OAUTH_EXPIRY_SKEW_SECONDS: Final = 300
ANTHROPIC_OAUTH_CALLBACK_TIMEOUT_SECONDS: Final = 300
ANTHROPIC_OAUTH_BRAND: Final = "Anthropic"
_DEFAULT_EXPIRES_IN: Final = 3600

_ANTHROPIC_OAUTH_REFRESH_LOCK: Final = threading.Lock()
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


def _default_token_dir() -> str:
    return os.path.expanduser("~/.config/litellm/anthropic_oauth")


def _json_headers() -> dict[str, str]:
    return {"Accept": "application/json", "Content-Type": "application/json"}  # mutable-ok: fresh per request


class AnthropicOAuthRecord(TypedDict):
    access_token: NotRequired[ReadOnly[str]]
    refresh_token: NotRequired[ReadOnly[str]]
    token_type: NotRequired[ReadOnly[str]]
    scope: NotRequired[ReadOnly[str | None]]
    expires_at: NotRequired[ReadOnly[float | None]]


class _TokenPayload(TypedDict):
    access_token: NotRequired[ReadOnly[str]]
    refresh_token: NotRequired[ReadOnly[str]]
    token_type: NotRequired[ReadOnly[str]]
    scope: NotRequired[ReadOnly[str]]
    expires_in: NotRequired[ReadOnly[float]]


_RECORD_ADAPTER: Final = TypeAdapter(AnthropicOAuthRecord)
_PAYLOAD_ADAPTER: Final = TypeAdapter(_TokenPayload)


class AnthropicOAuthError(Exception):
    pass


class AnthropicOAuthLoginRequiredError(AnthropicOAuthError):
    pass


def _login_hint(reason: str) -> AnthropicOAuthLoginRequiredError:
    return AnthropicOAuthLoginRequiredError(f"{reason} Run `litellm anthropic-oauth login`.")


def parse_authorization_input(value: str) -> tuple[str, str | None]:
    """Parse a pasted authorization code or redirect URL, as an Anthropic error."""
    try:
        return _parse_authorization_input(value)
    except ValueError as exc:
        raise AnthropicOAuthError(str(exc)) from exc


class AnthropicOAuthAuthenticator:
    """Obtains, stores and refreshes a Claude Pro/Max OAuth token."""

    def __init__(self, http_client: httpx.Client | HTTPHandler | None = None) -> None:
        token_dir: Final = get_secret_str("ANTHROPIC_OAUTH_TOKEN_DIR") or _default_token_dir()
        auth_file_name: Final = get_secret_str("ANTHROPIC_OAUTH_AUTH_FILE") or "auth.json"
        self.token_dir = token_dir
        self.auth_file = os.path.join(token_dir, auth_file_name)
        self.http_client = http_client
        self._store = TokenFile(self.auth_file)

    def get_access_token(self) -> str:
        """Return a usable access token, refreshing it first if it has expired."""
        auth_data: Final = self._read_auth_file()
        if not auth_data:
            raise _login_hint("Anthropic OAuth login required.")

        access_token = auth_data.get("access_token")
        if access_token and not self._is_expired(auth_data):
            return access_token

        if not auth_data.get("refresh_token"):
            raise _login_hint("Anthropic OAuth refresh token missing.")

        with _ANTHROPIC_OAUTH_REFRESH_LOCK:
            # Another thread may have refreshed while this one waited for the lock.
            locked_auth_data: Final = self._read_auth_file() or auth_data
            access_token = locked_auth_data.get("access_token")
            if access_token and not self._is_expired(locked_auth_data):
                return access_token
            refreshed: Final = self._refresh_tokens(locked_auth_data).get("access_token")
            if not refreshed:
                raise _login_hint("Anthropic OAuth refresh returned no access token.")
            return refreshed

    def login(
        self,
        force: bool = False,
        no_browser: bool = False,
        prompt_for_code: Callable[[str], str] | None = None,
    ) -> AnthropicOAuthRecord:
        """Run the authorization-code flow and store the result.

        `prompt_for_code` is for a host with no reachable browser: it is handed the authorize URL and returns
        what the user pasted back, and no loopback server is started.
        """
        existing: Final = self._read_auth_file()
        if existing and not force and existing.get("access_token"):
            if not self._is_expired(existing):
                return existing
            if existing.get("refresh_token"):
                try:
                    return self._refresh_tokens(existing)
                except AnthropicOAuthError:
                    pass  # fall through to a fresh login

        verifier, challenge = pkce_pair()
        state: Final = new_state()

        if prompt_for_code is not None:
            pending: Final = self.begin_login()
            return self.complete_login(prompt_for_code(pending["authorize_url"]))
        return self._login_by_loopback(verifier, challenge, state, no_browser)

    def begin_login(self) -> PendingLogin:
        """Start a login and persist what completing it will need.

        Returns the pending record, whose `authorize_url` is what the user opens. The PKCE verifier is
        written to disk so `complete_login` can run in a later invocation.
        """
        verifier, challenge = pkce_pair()
        state: Final = new_state()
        redirect_uri: Final = f"http://127.0.0.1:{ANTHROPIC_OAUTH_REDIRECT_PORT}{ANTHROPIC_OAUTH_REDIRECT_PATH}"
        pending: Final[PendingLogin] = {  # mutable-ok: written straight to the pending file
            "verifier": verifier,
            "state": state,
            "redirect_uri": redirect_uri,
            "authorize_url": self._build_authorize_url(redirect_uri, challenge, state),
            "token_endpoint": ANTHROPIC_OAUTH_TOKEN_URL,
            "created_at": time.time(),
        }
        self._pending_file().write(pending)
        return pending

    def complete_login(self, pasted: str) -> AnthropicOAuthRecord:
        """Finish a login started by `begin_login`, from a pasted code or redirect URL."""
        store: Final = self._pending_file()
        raw: Final = store.read()
        if not isinstance(raw, dict) or not raw.get("verifier"):
            raise AnthropicOAuthError("No Anthropic login is in progress. Run `litellm anthropic-oauth login` first.")
        if not pending_is_fresh(raw):
            store.delete()
            raise AnthropicOAuthError("The Anthropic login has expired. Run `litellm anthropic-oauth login` again.")

        code, returned_state = parse_authorization_input(pasted)
        if returned_state is not None and returned_state != raw.get("state"):
            raise AnthropicOAuthError(
                "This code came from a different login attempt. Run `litellm anthropic-oauth login` again and "
                "use the URL it prints."
            )

        record: Final = self._exchange_and_store(
            code=code,
            state=raw["state"],
            verifier=raw["verifier"],
            redirect_uri=raw["redirect_uri"],
        )
        store.delete()
        return record

    def _pending_file(self) -> TokenFile:
        return TokenFile(os.path.join(self.token_dir, "pending.json"))

    def logout(self) -> bool:
        """Delete the stored credentials. Returns whether a file was removed."""
        self._pending_file().delete()
        return self._store.delete()

    def _login_by_loopback(
        self,
        verifier: str,
        challenge: str,
        state: str,
        no_browser: bool,
    ) -> AnthropicOAuthRecord:
        try:
            server, redirect_uri = start_callback_server(
                expected_state=state,
                preferred_port=ANTHROPIC_OAUTH_REDIRECT_PORT,
                callback_path=ANTHROPIC_OAUTH_REDIRECT_PATH,
                brand=ANTHROPIC_OAUTH_BRAND,
            )
        except OSError as exc:
            raise AnthropicOAuthError(str(exc)) from exc

        authorize_url: Final = self._build_authorize_url(redirect_uri, challenge, state)
        if no_browser or not webbrowser.open(authorize_url):
            sys.stdout.write(f"Open this URL to authenticate with Anthropic:\n{authorize_url}\n")
            sys.stdout.flush()

        result: Final = wait_for_callback(server, ANTHROPIC_OAUTH_CALLBACK_TIMEOUT_SECONDS)
        if result is None:
            raise AnthropicOAuthError("Timed out waiting for Anthropic OAuth callback")
        if result["state"] != state:
            raise AnthropicOAuthError("Anthropic OAuth state mismatch")
        if result["error"]:
            raise AnthropicOAuthError(
                f"Anthropic authorization failed: {result['error_description'] or result['error']}"
            )
        code: Final = result["code"]
        if not code:
            raise AnthropicOAuthError("Anthropic authorization failed: no code returned")

        return self._exchange_and_store(code=code, state=state, verifier=verifier, redirect_uri=redirect_uri)

    def _exchange_and_store(
        self,
        code: str,
        state: str,
        verifier: str,
        redirect_uri: str,
    ) -> AnthropicOAuthRecord:
        payload: Final = self._post_token(
            {  # mutable-ok: request body, built once and handed straight to the client
                "grant_type": "authorization_code",
                "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            }
        )
        record: Final = self._build_auth_record(payload)
        self._store.write(record)
        return record

    def _client(self) -> httpx.Client | HTTPHandler:
        return self.http_client or _get_httpx_client()

    def _read_auth_file(self) -> AnthropicOAuthRecord | None:
        data: Final = self._store.read()
        if data is None:
            return None  # no login yet, which is not a problem worth logging
        try:
            return _RECORD_ADAPTER.validate_python(data)
        except ValidationError:
            verbose_logger.warning("Ignoring unreadable Anthropic OAuth record at %s", self.auth_file)
            return None

    def _is_expired(self, auth_data: AnthropicOAuthRecord) -> bool:
        expires_at: Final = auth_data.get("expires_at")
        if expires_at is None:
            return True
        try:
            return time.time() >= float(expires_at) - ANTHROPIC_OAUTH_EXPIRY_SKEW_SECONDS
        except (TypeError, ValueError):
            return True

    def _build_authorize_url(self, redirect_uri: str, challenge: str, state: str) -> str:
        params: Final = {  # mutable-ok: query parameters, urlencoded immediately below
            "code": "true",
            "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": ANTHROPIC_OAUTH_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        return f"{ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urlencode(params)}"

    def _post_token(self, body: dict[str, str]) -> _TokenPayload:
        try:
            response: Final = self._client().post(
                ANTHROPIC_OAUTH_TOKEN_URL,
                headers=_json_headers(),
                json=body,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AnthropicOAuthError(
                f"Anthropic OAuth token request failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AnthropicOAuthError(f"Anthropic OAuth token request failed: {exc}") from exc
        try:
            payload: Final = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise AnthropicOAuthError("Anthropic OAuth token response was not valid JSON") from exc
        try:
            return _PAYLOAD_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            raise AnthropicOAuthError(f"Anthropic OAuth token response had unexpected fields: {exc}") from exc

    def _build_auth_record(
        self,
        payload: _TokenPayload,
        fallback_refresh_token: str | None = None,
    ) -> AnthropicOAuthRecord:
        access_token: Final = payload.get("access_token")
        refresh_token: Final = payload.get("refresh_token") or fallback_refresh_token
        if not access_token:
            raise AnthropicOAuthError("Anthropic OAuth token response missing access_token")
        if not refresh_token:
            raise AnthropicOAuthError("Anthropic OAuth token response missing refresh_token")
        try:
            expires_at = int(time.time() + int(payload.get("expires_in") or _DEFAULT_EXPIRES_IN))
        except (TypeError, ValueError):
            expires_at = int(time.time() + _DEFAULT_EXPIRES_IN)
        return {  # mutable-ok: the credential record, written verbatim to the token file
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": payload.get("token_type") or "Bearer",
            "scope": payload.get("scope"),
            "expires_at": expires_at,
        }

    def _refresh_tokens(self, auth_data: AnthropicOAuthRecord) -> AnthropicOAuthRecord:
        refresh_token: Final = auth_data.get("refresh_token")
        if not refresh_token:
            raise _login_hint("Anthropic OAuth refresh token missing.")
        payload: Final = self._post_token(
            {  # mutable-ok: request body, built once and handed straight to the client
                "grant_type": "refresh_token",
                "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
                "refresh_token": refresh_token,
            }
        )
        refreshed: Final = self._build_auth_record(payload, fallback_refresh_token=refresh_token)
        self._store.write(refreshed)
        return refreshed


def should_use_anthropic_oauth(litellm_params: Mapping[str, object] | None) -> bool:
    return bool((litellm_params or _NO_PARAMS).get("use_anthropic_oauth"))
