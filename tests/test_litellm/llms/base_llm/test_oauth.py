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
    TokenFile,
    new_state,
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
