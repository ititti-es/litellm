import json
import subprocess
import sys

import pytest

from litellm.llms.base_llm.oauth import (
    TokenFile,
    credential_store_for,
    set_credential_store_factory,
)
from litellm.proxy.oauth_credential_store import (
    _decrypted,
    _encrypted,
    _lock_key,
    credential_name,
    use_database_oauth_credentials,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    yield
    set_credential_store_factory(None)


@pytest.fixture
def salt(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", "a-test-salt-key")


# --- naming and locking ----------------------------------------------------------------------------


def test_credentials_are_namespaced_so_they_cannot_collide_with_a_users_own():
    assert credential_name("anthropic") == "oauth/anthropic"


def test_each_provider_gets_its_own_lock():
    """One shared key would serialise refreshes that have nothing to do with each other."""
    assert _lock_key("anthropic") != _lock_key("xai")


def test_the_lock_key_is_stable_across_processes():
    """`hash()` on a string is randomised per process, so two replicas would never meet on the same key."""
    code = "from litellm.proxy.oauth_credential_store import _lock_key; print(_lock_key('anthropic'))"
    first = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin"}
    ).stdout.strip()
    second = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONHASHSEED": "2", "PATH": "/usr/bin"}
    ).stdout.strip()

    assert first and first == second


def test_the_lock_key_fits_a_postgres_bigint():
    for provider in ("anthropic", "xai", "chatgpt"):
        assert 0 < _lock_key(provider) < 2**63


# --- encryption at rest ----------------------------------------------------------------------------


def test_a_record_survives_encryption_and_decryption(salt):
    record = {"access_token": "sk-ant-oat01-secret", "refresh_token": "refresh", "expires_at": 1789925329}

    assert _decrypted(_encrypted(record)) == record


def test_no_field_is_stored_in_the_clear(salt):
    record = {"access_token": "sk-ant-oat01-secret", "refresh_token": "refresh-secret", "expires_at": 123}
    stored = _encrypted(record)

    serialised = json.dumps(stored)
    assert "sk-ant-oat01-secret" not in serialised
    assert "refresh-secret" not in serialised
    assert all(isinstance(value, str) for value in stored.values())


def test_non_string_fields_keep_their_type(salt):
    """expires_at is compared numerically; coming back as a string would make every token look expired."""
    restored = _decrypted(_encrypted({"expires_at": 1789925329, "scope": None}))

    assert restored["expires_at"] == 1789925329
    assert restored["scope"] is None


def test_a_record_that_cannot_be_decrypted_reads_as_absent(salt):
    """A wrong salt key must look like "not logged in", not like a corrupt token."""
    assert _decrypted({"access_token": "not-actually-encrypted-%%%"}) is None


def test_a_record_of_the_wrong_shape_reads_as_absent(salt):
    assert _decrypted({"access_token": 12345}) is None


# --- choosing the store ----------------------------------------------------------------------------


def test_files_are_used_when_there_is_no_database(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert use_database_oauth_credentials(None) is False
    assert isinstance(credential_store_for("anthropic", str(tmp_path / "a.json")), TokenFile)


def test_the_database_is_used_when_one_is_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost:5432/db")

    assert use_database_oauth_credentials(None) is True
    assert not isinstance(credential_store_for("anthropic", str(tmp_path / "a.json")), TokenFile)


def test_files_can_be_kept_even_with_a_database(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost:5432/db")

    assert use_database_oauth_credentials(False) is False
    assert isinstance(credential_store_for("anthropic", str(tmp_path / "a.json")), TokenFile)


def test_asking_for_the_database_without_one_falls_back_to_files(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    assert use_database_oauth_credentials(True) is False
    assert isinstance(credential_store_for("anthropic", str(tmp_path / "a.json")), TokenFile)


def test_a_store_that_cannot_be_built_falls_back_to_a_file(tmp_path):
    """A credential store failing to construct must not make every login impossible."""

    def broken(provider: str, path: str):
        raise RuntimeError("no database today")

    set_credential_store_factory(broken)

    assert isinstance(credential_store_for("anthropic", str(tmp_path / "a.json")), TokenFile)


# --- the file store still satisfies the protocol -----------------------------------------------------


def test_the_file_store_guard_is_a_usable_context_manager(tmp_path):
    store = TokenFile(str(tmp_path / "auth.json"))

    with store.guard():
        store.write({"access_token": "x"})

    assert store.read() == {"access_token": "x"}
