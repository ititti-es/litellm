import json
import time
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import litellm
from litellm.llms.anthropic.oauth import (
    ANTHROPIC_OAUTH_AUTHORIZE_URL,
    ANTHROPIC_OAUTH_CLIENT_ID,
    ANTHROPIC_OAUTH_SCOPE,
    ANTHROPIC_OAUTH_TOKEN_URL,
    AnthropicOAuthAuthenticator,
    AnthropicOAuthError,
    AnthropicOAuthLoginRequiredError,
    parse_authorization_input,
    should_use_anthropic_oauth,
)


class _StubClient:
    """Stands in for the httpx client, recording each token request."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def post(self, url, headers=None, json=None, **kwargs):
        self.requests.append({"url": url, "headers": headers or {}, "json": json or {}})
        if not self.responses:
            raise AssertionError("no stubbed response left")
        return self.responses.pop(0)


def _token_response(status=200, **overrides):
    body = {
        "access_token": "sk-ant-oat01-new",
        "refresh_token": "refresh-new",
        "token_type": "Bearer",
        "scope": "user:inference",
        "expires_in": 3600,
    }
    body.update(overrides)
    return httpx.Response(status_code=status, json=body, request=httpx.Request("POST", ANTHROPIC_OAUTH_TOKEN_URL))


def _authenticator(tmp_path, monkeypatch, client=None):
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN_DIR", str(tmp_path / "anthropic_oauth"))
    return AnthropicOAuthAuthenticator(http_client=client)


def _write_record(tmp_path, payload):
    token_dir = tmp_path / "anthropic_oauth"
    token_dir.mkdir(exist_ok=True)
    (token_dir / "auth.json").write_text(json.dumps(payload))
    return token_dir


# --- parsing what a user pastes back -------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("abc123#state-1", ("abc123", "state-1")),
        ("  abc123#state-1  ", ("abc123", "state-1")),
        ("abc123", ("abc123", None)),
        ("http://127.0.0.1:53692/callback?code=abc123&state=state-1", ("abc123", "state-1")),
        ("https://example.com/callback?code=abc123", ("abc123", None)),
    ],
)
def test_parse_authorization_input(value, expected):
    assert parse_authorization_input(value) == expected


def test_parse_authorization_input_rejects_empty():
    with pytest.raises(AnthropicOAuthError):
        parse_authorization_input("   ")


def test_parse_authorization_input_rejects_url_without_code():
    with pytest.raises(AnthropicOAuthError):
        parse_authorization_input("http://127.0.0.1:53692/callback?state=only")


# --- the authorize URL ----------------------------------------------------------------------------


def test_authorize_url_carries_pkce_and_scopes(tmp_path, monkeypatch):
    url = _authenticator(tmp_path, monkeypatch)._build_authorize_url(
        "http://127.0.0.1:53692/callback", "challenge-value", "state-value"
    )
    parsed = urlparse(url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    assert url.startswith(ANTHROPIC_OAUTH_AUTHORIZE_URL)
    assert params["client_id"] == ANTHROPIC_OAUTH_CLIENT_ID
    assert params["response_type"] == "code"
    assert params["code_challenge"] == "challenge-value"
    assert params["code_challenge_method"] == "S256"
    assert params["state"] == "state-value"
    assert params["scope"] == ANTHROPIC_OAUTH_SCOPE
    assert "user:inference" in params["scope"]


# --- token exchange and refresh --------------------------------------------------------------------


def test_exchange_posts_json_with_state_and_verifier(tmp_path, monkeypatch):
    client = _StubClient(_token_response())
    authenticator = _authenticator(tmp_path, monkeypatch, client)

    record = authenticator._exchange_and_store(
        code="the-code",
        state="the-state",
        verifier="the-verifier",
        redirect_uri="http://127.0.0.1:53692/callback",
    )

    sent = client.requests[0]
    assert sent["url"] == ANTHROPIC_OAUTH_TOKEN_URL
    # Anthropic's token endpoint takes JSON and echoes state back, unlike the form-encoded xAI flow.
    assert sent["headers"]["Content-Type"] == "application/json"
    assert sent["json"] == {
        "grant_type": "authorization_code",
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "code": "the-code",
        "state": "the-state",
        "redirect_uri": "http://127.0.0.1:53692/callback",
        "code_verifier": "the-verifier",
    }
    assert record["access_token"] == "sk-ant-oat01-new"
    assert record["refresh_token"] == "refresh-new"


def test_exchange_persists_with_owner_only_permissions(tmp_path, monkeypatch):
    import os
    import stat

    authenticator = _authenticator(tmp_path, monkeypatch, _StubClient(_token_response()))
    authenticator._exchange_and_store(
        code="c", state="s", verifier="v", redirect_uri="http://127.0.0.1:53692/callback"
    )

    mode = stat.S_IMODE(os.stat(authenticator.auth_file).st_mode)
    assert mode == 0o600, f"credential file should be owner-only, got {oct(mode)}"
    assert json.loads(open(authenticator.auth_file).read())["access_token"] == "sk-ant-oat01-new"


def test_access_token_returns_stored_token_while_fresh(tmp_path, monkeypatch):
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-fresh", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    client = _StubClient()
    assert _authenticator(tmp_path, monkeypatch, client).get_access_token() == "sk-ant-oat01-fresh"
    assert client.requests == [], "a fresh token must not trigger a refresh"


def test_access_token_refreshes_when_expired(tmp_path, monkeypatch):
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-old", "refresh_token": "refresh-old", "expires_at": time.time() - 10},
    )
    client = _StubClient(_token_response())

    assert _authenticator(tmp_path, monkeypatch, client).get_access_token() == "sk-ant-oat01-new"
    assert client.requests[0]["json"] == {
        "grant_type": "refresh_token",
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "refresh_token": "refresh-old",
    }


def test_access_token_refreshes_inside_the_expiry_skew(tmp_path, monkeypatch):
    # Still valid by the clock, but close enough that a request could outlive it.
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-old", "refresh_token": "refresh-old", "expires_at": time.time() + 30},
    )
    client = _StubClient(_token_response())
    assert _authenticator(tmp_path, monkeypatch, client).get_access_token() == "sk-ant-oat01-new"


def test_refresh_keeps_previous_refresh_token_when_none_returned(tmp_path, monkeypatch):
    _write_record(
        tmp_path,
        {"access_token": "old", "refresh_token": "refresh-keep", "expires_at": time.time() - 10},
    )
    response = _token_response()
    body = json.loads(response.content)
    del body["refresh_token"]
    client = _StubClient(
        httpx.Response(200, json=body, request=httpx.Request("POST", ANTHROPIC_OAUTH_TOKEN_URL))
    )
    authenticator = _authenticator(tmp_path, monkeypatch, client)

    authenticator.get_access_token()
    assert json.loads(open(authenticator.auth_file).read())["refresh_token"] == "refresh-keep"


def test_access_token_without_credentials_asks_for_login(tmp_path, monkeypatch):
    with pytest.raises(AnthropicOAuthLoginRequiredError, match="anthropic-oauth login"):
        _authenticator(tmp_path, monkeypatch).get_access_token()


def test_access_token_without_refresh_token_asks_for_login(tmp_path, monkeypatch):
    _write_record(tmp_path, {"access_token": "old", "expires_at": time.time() - 10})
    with pytest.raises(AnthropicOAuthLoginRequiredError, match="anthropic-oauth login"):
        _authenticator(tmp_path, monkeypatch).get_access_token()


def test_token_error_surfaces_the_provider_response(tmp_path, monkeypatch):
    client = _StubClient(
        httpx.Response(
            400,
            json={"error": "invalid_grant"},
            request=httpx.Request("POST", ANTHROPIC_OAUTH_TOKEN_URL),
        )
    )
    with pytest.raises(AnthropicOAuthError, match="invalid_grant"):
        _authenticator(tmp_path, monkeypatch, client)._exchange_and_store(
            code="c", state="s", verifier="v", redirect_uri="http://127.0.0.1:53692/callback"
        )


def test_missing_access_token_in_response_is_rejected(tmp_path, monkeypatch):
    client = _StubClient(
        httpx.Response(
            200,
            json={"refresh_token": "r"},
            request=httpx.Request("POST", ANTHROPIC_OAUTH_TOKEN_URL),
        )
    )
    with pytest.raises(AnthropicOAuthError, match="missing access_token"):
        _authenticator(tmp_path, monkeypatch, client)._exchange_and_store(
            code="c", state="s", verifier="v", redirect_uri="http://127.0.0.1:53692/callback"
        )


# --- login orchestration ----------------------------------------------------------------------------


def test_login_reuses_a_valid_stored_token(tmp_path, monkeypatch):
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-fresh", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    client = _StubClient()
    assert _authenticator(tmp_path, monkeypatch, client).login()["access_token"] == "sk-ant-oat01-fresh"
    assert client.requests == []


def test_manual_login_rejects_a_mismatched_state(tmp_path, monkeypatch):
    authenticator = _authenticator(tmp_path, monkeypatch, _StubClient(_token_response()))
    with pytest.raises(AnthropicOAuthError, match="state mismatch"):
        authenticator.login(prompt_for_code=lambda url: "the-code#not-the-state")


def test_manual_login_exchanges_a_pasted_code(tmp_path, monkeypatch):
    client = _StubClient(_token_response())
    authenticator = _authenticator(tmp_path, monkeypatch, client)
    seen: list[str] = []

    def paste(url: str) -> str:
        seen.append(url)
        # Echo back the state the authorize URL asked for, as the browser would.
        return "the-code#" + parse_qs(urlparse(url).query)["state"][0]

    record = authenticator.login(prompt_for_code=paste)

    assert record["access_token"] == "sk-ant-oat01-new"
    assert seen[0].startswith(ANTHROPIC_OAUTH_AUTHORIZE_URL)
    assert client.requests[0]["json"]["code"] == "the-code"


def test_logout_removes_the_credentials(tmp_path, monkeypatch):
    _write_record(tmp_path, {"access_token": "a", "refresh_token": "r", "expires_at": time.time() + 10})
    authenticator = _authenticator(tmp_path, monkeypatch)

    assert authenticator.logout() is True
    assert authenticator.logout() is False, "a second logout has nothing left to remove"


# --- wiring into the request path ----------------------------------------------------------------------


def test_should_use_anthropic_oauth_reads_the_deployment_flag():
    assert should_use_anthropic_oauth({"use_anthropic_oauth": True}) is True
    assert should_use_anthropic_oauth({"use_anthropic_oauth": False}) is False
    assert should_use_anthropic_oauth({}) is False
    assert should_use_anthropic_oauth(None) is False


def test_validate_environment_uses_the_oauth_token(tmp_path, monkeypatch):
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-live", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN_DIR", str(tmp_path / "anthropic_oauth"))

    headers = AnthropicConfig().validate_environment(
        headers={},
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={"use_anthropic_oauth": True},
    )

    assert headers["authorization"] == "Bearer sk-ant-oat01-live"
    assert "x-api-key" not in headers
    assert "oauth-2025-04-20" in headers["anthropic-beta"]


def test_explicit_api_key_wins_over_oauth(tmp_path, monkeypatch):
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-live", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN_DIR", str(tmp_path / "anthropic_oauth"))

    headers = AnthropicConfig().validate_environment(
        headers={},
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={"use_anthropic_oauth": True},
        api_key="sk-ant-api03-explicit",
    )

    assert headers.get("x-api-key") == "sk-ant-api03-explicit"
    assert "authorization" not in headers


def test_missing_login_surfaces_as_an_authentication_error(tmp_path, monkeypatch):
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN_DIR", str(tmp_path / "empty"))

    with pytest.raises(litellm.AuthenticationError, match="anthropic-oauth login"):
        AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={"use_anthropic_oauth": True},
        )


def test_oauth_is_not_consulted_without_the_flag(tmp_path, monkeypatch):
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    _write_record(
        tmp_path,
        {"access_token": "sk-ant-oat01-live", "refresh_token": "r", "expires_at": time.time() + 3600},
    )
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN_DIR", str(tmp_path / "anthropic_oauth"))

    with pytest.raises(litellm.AuthenticationError, match="Missing Anthropic API Key"):
        AnthropicConfig().validate_environment(
            headers={},
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
        )
