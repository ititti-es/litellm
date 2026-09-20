import base64
import hashlib
import json
import os
import stat
import threading
import time
import urllib.request

import pytest

from litellm.llms.base_llm.oauth import (
    PENDING_LOGIN_TTL_SECONDS,
    TokenFile,
    is_remote_session,
    pending_is_fresh,
    new_state,
    parse_authorization_input,
    pkce_pair,
    start_callback_server,
    wait_for_callback,
)


# --- PKCE ---------------------------------------------------------------------------------------


def test_pkce_challenge_is_the_s256_of_the_verifier():
    verifier, challenge = pkce_pair()
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected


def test_pkce_values_are_url_safe_and_unpadded():
    verifier, challenge = pkce_pair()
    for value in (verifier, challenge):
        assert "=" not in value and "+" not in value and "/" not in value


def test_pkce_pairs_are_unique():
    assert len({pkce_pair()[0] for _ in range(50)}) == 50


def test_states_are_unique():
    assert len({new_state() for _ in range(50)}) == 50


# --- the loopback callback server ---------------------------------------------------------------


def _get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _serve_once(server):
    thread = threading.Thread(target=lambda: wait_for_callback(server, 10), daemon=True)
    thread.start()
    return thread


def test_callback_server_captures_the_code():
    server, redirect_uri = start_callback_server("state-1", 0, "/callback", "Test")
    thread = _serve_once(server)

    status, body = _get(f"{redirect_uri}?code=the-code&state=state-1")
    thread.join(timeout=5)

    assert status == 200
    assert "authorization received" in body.lower()
    assert server.callback_result is not None
    assert server.callback_result["code"] == "the-code"
    assert server.callback_result["state"] == "state-1"


def test_callback_server_rejects_a_mismatched_state():
    server, redirect_uri = start_callback_server("state-1", 0, "/callback", "Test")
    thread = _serve_once(server)

    status, body = _get(f"{redirect_uri}?code=the-code&state=forged")
    thread.join(timeout=5)

    assert status == 400
    assert "state mismatch" in body.lower()


def test_callback_server_reports_provider_errors():
    server, redirect_uri = start_callback_server("state-1", 0, "/callback", "Test")
    thread = _serve_once(server)

    status, body = _get(f"{redirect_uri}?error=access_denied&error_description=nope&state=state-1")
    thread.join(timeout=5)

    assert status == 200
    assert "authorization failed" in body.lower()
    assert server.callback_result["error"] == "access_denied"
    assert server.callback_result["error_description"] == "nope"


def test_callback_server_ignores_other_paths():
    server, redirect_uri = start_callback_server("state-1", 0, "/callback", "Test")
    base = redirect_uri.rsplit("/callback", 1)[0]
    thread = threading.Thread(target=lambda: wait_for_callback(server, 3), daemon=True)
    thread.start()

    status, _ = _get(f"{base}/favicon.ico")
    assert status == 404
    assert server.callback_result is None
    thread.join(timeout=5)


def test_wait_for_callback_returns_none_on_timeout():
    server, _ = start_callback_server("state-1", 0, "/callback", "Test")
    assert wait_for_callback(server, 0.1) is None


def test_busy_preferred_port_falls_back_to_an_ephemeral_one():
    first, first_uri = start_callback_server("state-1", 0, "/callback", "Test")
    busy_port = first.server_address[1]
    try:
        second, second_uri = start_callback_server("state-2", busy_port, "/callback", "Test")
        try:
            assert second.server_address[1] != busy_port
            assert second_uri != first_uri
        finally:
            second.server_close()
    finally:
        first.server_close()


# --- the credential file --------------------------------------------------------------------------


def test_write_creates_an_owner_only_file_and_directory(tmp_path):
    store = TokenFile(str(tmp_path / "creds" / "auth.json"))
    store.write({"access_token": "secret"})

    assert stat.S_IMODE(os.stat(store.path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(store.directory).st_mode) == 0o700
    assert store.read() == {"access_token": "secret"}


def test_write_replaces_the_previous_record(tmp_path):
    store = TokenFile(str(tmp_path / "auth.json"))
    store.write({"access_token": "first"})
    store.write({"access_token": "second"})

    assert store.read() == {"access_token": "second"}


def test_write_leaves_no_temporary_files_behind(tmp_path):
    store = TokenFile(str(tmp_path / "auth.json"))
    store.write({"access_token": "secret"})

    assert [p.name for p in tmp_path.iterdir()] == ["auth.json"]


def test_read_of_a_missing_file_is_none(tmp_path):
    assert TokenFile(str(tmp_path / "absent.json")).read() is None


def test_read_of_a_corrupt_file_is_none(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json")
    assert TokenFile(str(path)).read() is None


def test_delete_reports_whether_anything_was_removed(tmp_path):
    store = TokenFile(str(tmp_path / "auth.json"))
    store.write({"access_token": "secret"})

    assert store.delete() is True
    assert store.delete() is False
    assert store.read() is None


def test_concurrent_writers_never_leave_a_partial_record(tmp_path):
    """A reader must see one complete record, never a half-written one."""
    store = TokenFile(str(tmp_path / "auth.json"))
    store.write({"access_token": "initial", "padding": "x" * 10_000})
    errors: list[Exception] = []

    def writer(index: int):
        try:
            for _ in range(20):
                store.write({"access_token": f"token-{index}", "padding": "x" * 10_000})
        except Exception as exc:  # pragma: no cover - a failure here is the point of the test
            errors.append(exc)

    def reader():
        try:
            for _ in range(50):
                record = store.read()
                assert record is not None, "the record must never vanish mid-write"
                assert record["access_token"].startswith(("initial", "token-"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
    threads.append(threading.Thread(target=reader))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == []
    assert store.read() is not None


# --- remote sessions ------------------------------------------------------------------------------


def _clear_session_env(monkeypatch):
    for name in ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("variable", ["SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"])
def test_ssh_environment_counts_as_remote(variable, monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv(variable, "something")
    assert is_remote_session() is True


def test_a_local_desktop_session_is_not_remote(monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("DISPLAY", ":0")
    assert is_remote_session() is False


def test_a_headless_linux_box_counts_as_remote(monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setattr("sys.platform", "linux")
    assert is_remote_session() is True


def test_wayland_counts_as_a_usable_display(monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert is_remote_session() is False


# --- pasted authorization input -------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("abc123#state-1", ("abc123", "state-1")),
        ("  abc123#state-1  ", ("abc123", "state-1")),
        ("abc123", ("abc123", None)),
        ("http://127.0.0.1:56121/callback?code=abc123&state=state-1", ("abc123", "state-1")),
        ("http://127.0.0.1:56121/callback?code=abc123", ("abc123", None)),
        ("http://127.0.0.1:56121/callback?state=state-1&code=abc123", ("abc123", "state-1")),
    ],
)
def test_parse_authorization_input(value, expected):
    assert parse_authorization_input(value) == expected


@pytest.mark.parametrize("value", ["", "   ", "http://127.0.0.1:56121/callback?state=only"])
def test_parse_authorization_input_rejects_unusable_values(value):
    with pytest.raises(ValueError):
        parse_authorization_input(value)


# --- an authorization URL pasted back by mistake ----------------------------------------------------


ANTHROPIC_AUTHORIZE_URL = (
    "https://claude.ai/oauth/authorize?code=true&client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e"
    "&response_type=code&redirect_uri=http%3A%2F%2F127.0.0.1%3A53692%2Fcallback"
    "&scope=org%3Acreate_api_key+user%3Ainference&code_challenge=-xqMYIjx8znu"
    "&code_challenge_method=S256&state=1234cd1d3bbb4411be56b98f49e9c54c"
)

XAI_AUTHORIZE_URL = (
    "https://auth.x.ai/oauth2/authorize?response_type=code&client_id=b1a00492"
    "&redirect_uri=http%3A%2F%2F127.0.0.1%3A56121%2Fcallback&scope=openid+profile"
    "&code_challenge=PO8n&code_challenge_method=S256&state=6d1723b4&nonce=6287033b"
)


@pytest.mark.parametrize("url", [ANTHROPIC_AUTHORIZE_URL, XAI_AUTHORIZE_URL])
def test_pasting_the_authorization_url_back_is_rejected(url):
    """Anthropic's authorize URL carries a literal `code=true`, which would otherwise be read as the code."""
    with pytest.raises(ValueError, match="authorization URL, not the result"):
        parse_authorization_input(url)


def test_the_rejection_names_what_to_copy_instead():
    with pytest.raises(ValueError, match="127.0.0.1 page it then fails to load"):
        parse_authorization_input(ANTHROPIC_AUTHORIZE_URL)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:53692/callback?code=REAL&state=st",
        "http://127.0.0.1:56121/callback?code=REAL",
        "http://localhost:53692/callback?state=st&code=REAL",
    ],
)
def test_a_real_redirect_url_is_still_accepted(url):
    code, _ = parse_authorization_input(url)
    assert code == "REAL"


# --- pending logins -------------------------------------------------------------------------------


def _pending(**overrides):
    record = {
        "verifier": "v",
        "state": "s",
        "redirect_uri": "http://127.0.0.1:1/callback",
        "authorize_url": "https://example.com/authorize",
        "token_endpoint": "https://example.com/token",
        "created_at": time.time(),
    }
    record.update(overrides)
    return record


def test_a_fresh_pending_login_is_usable():
    assert pending_is_fresh(_pending()) is True


def test_an_old_pending_login_is_not_usable():
    assert pending_is_fresh(_pending(created_at=time.time() - PENDING_LOGIN_TTL_SECONDS - 1)) is False


def test_a_pending_login_without_a_timestamp_is_not_usable():
    assert pending_is_fresh(_pending(created_at=None)) is False
    assert pending_is_fresh(_pending(created_at="lunchtime")) is False


# --- advertising a different host than the one bound --------------------------------------------------


def test_the_redirect_host_can_differ_from_the_bound_address():
    """A provider matches the redirect URI as a string; some register `localhost`, not `127.0.0.1`."""
    server, redirect_uri = start_callback_server("state-1", 0, "/callback", "Test", redirect_host="localhost")
    try:
        assert redirect_uri.startswith("http://localhost:")
        assert server.server_address[0] == "127.0.0.1", "the socket still binds the numeric loopback"
    finally:
        server.server_close()


def test_a_busy_port_fails_when_no_other_port_would_be_accepted():
    """Falling back to an arbitrary port would only produce a redirect URI the provider rejects."""
    first, _ = start_callback_server("state-1", 0, "/callback", "Test")
    busy_port = first.server_address[1]
    try:
        with pytest.raises(OSError, match=f"port {busy_port}"):
            start_callback_server("state-2", busy_port, "/callback", "Test", allow_ephemeral=False)
    finally:
        first.server_close()
