import json
import time

import click
import pytest

from litellm.proxy.oauth_cli import handle_oauth_command


@pytest.fixture
def isolated_credentials(tmp_path, monkeypatch):
    """Point every provider's credential file at a temporary directory."""
    for env in ("ANTHROPIC_OAUTH_TOKEN_DIR", "XAI_OAUTH_TOKEN_DIR", "CHATGPT_TOKEN_DIR"):
        monkeypatch.setenv(env, str(tmp_path / env.lower()))
    return tmp_path


def _write(tmp_path, env_dir, payload):
    directory = tmp_path / env_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "auth.json").write_text(json.dumps(payload))
    return directory / "auth.json"


# --- dispatch -------------------------------------------------------------------------------------


def test_unknown_provider_is_not_handled(isolated_credentials):
    assert handle_oauth_command(("serve", "--port", "4000")) is False


def test_too_few_arguments_are_not_handled(isolated_credentials):
    assert handle_oauth_command(("anthropic-oauth",)) is False
    assert handle_oauth_command(()) is False


def test_unknown_action_is_a_usage_error(isolated_credentials):
    with pytest.raises(click.UsageError, match="Unknown action 'frobnicate'"):
        handle_oauth_command(("anthropic-oauth", "frobnicate"))


@pytest.mark.parametrize("provider", ["anthropic-oauth", "xai-oauth", "chatgpt"])
def test_status_is_handled_for_every_provider(provider, isolated_credentials, capsys):
    assert handle_oauth_command((provider, "status")) is True
    assert "not logged in" in capsys.readouterr().out


# --- status reporting -------------------------------------------------------------------------------


def test_status_without_credentials_names_the_login_command(isolated_credentials, capsys):
    handle_oauth_command(("anthropic-oauth", "status"))
    assert "litellm anthropic-oauth login" in capsys.readouterr().out


def test_status_reports_a_stored_token_and_its_expiry(isolated_credentials, capsys):
    _write(
        isolated_credentials,
        "anthropic_oauth_token_dir",
        {"access_token": "sk-ant-oat01-x", "expires_at": time.time() + 3600, "scope": "user:inference"},
    )
    handle_oauth_command(("anthropic-oauth", "status"))

    out = capsys.readouterr().out
    assert "logged in" in out
    assert "user:inference" in out
    assert "in 59m" in out or "in 60m" in out
    assert "sk-ant-oat01-x" not in out, "a status line must not print the token itself"


def test_status_marks_an_expired_token(isolated_credentials, capsys):
    _write(
        isolated_credentials,
        "anthropic_oauth_token_dir",
        {"access_token": "sk-ant-oat01-x", "expires_at": time.time() - 60},
    )
    handle_oauth_command(("anthropic-oauth", "status"))
    assert "(expired)" in capsys.readouterr().out


def test_status_tolerates_a_corrupt_credential_file(isolated_credentials, capsys):
    directory = isolated_credentials / "anthropic_oauth_token_dir"
    directory.mkdir(parents=True)
    (directory / "auth.json").write_text("{not json")

    handle_oauth_command(("anthropic-oauth", "status"))
    assert "not logged in" in capsys.readouterr().out


def test_status_reports_the_chatgpt_account(isolated_credentials, capsys):
    _write(
        isolated_credentials,
        "chatgpt_token_dir",
        {"access_token": "tok", "expires_at": time.time() + 600, "account_id": "acct-123"},
    )
    handle_oauth_command(("chatgpt", "status"))
    assert "acct-123" in capsys.readouterr().out


# --- logout ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,env_dir",
    [
        ("anthropic-oauth", "anthropic_oauth_token_dir"),
        ("xai-oauth", "xai_oauth_token_dir"),
        ("chatgpt", "chatgpt_token_dir"),
    ],
)
def test_logout_removes_stored_credentials(provider, env_dir, isolated_credentials, capsys):
    path = _write(isolated_credentials, env_dir, {"access_token": "tok", "expires_at": time.time() + 600})

    assert handle_oauth_command((provider, "logout")) is True
    assert "credentials removed" in capsys.readouterr().out
    assert not path.exists()

    handle_oauth_command((provider, "logout"))
    assert "no stored credentials" in capsys.readouterr().out


# --- login ---------------------------------------------------------------------------------------------


def test_login_returns_a_valid_stored_token_without_a_browser(isolated_credentials, capsys):
    _write(
        isolated_credentials,
        "anthropic_oauth_token_dir",
        {"access_token": "sk-ant-oat01-x", "refresh_token": "r", "expires_at": time.time() + 3600},
    )

    assert handle_oauth_command(("anthropic-oauth", "login")) is True
    out = capsys.readouterr().out
    assert "login successful" in out
    assert "sk-ant-oat01-x" not in out, "a login summary must not print the token itself"


def test_login_failure_becomes_a_click_exception(isolated_credentials, monkeypatch):
    """A failed login exits cleanly rather than dumping a traceback at an operator."""
    from litellm.llms.anthropic import oauth as anthropic_oauth

    def explode(self, **kwargs):
        raise anthropic_oauth.AnthropicOAuthError("authorization server said no")

    monkeypatch.setattr(anthropic_oauth.AnthropicOAuthAuthenticator, "login", explode)

    with pytest.raises(click.ClickException, match="authorization server said no"):
        handle_oauth_command(("anthropic-oauth", "login"))


# --- remote sessions choose the paste flow ------------------------------------------------------------


@pytest.fixture
def local_desktop(monkeypatch):
    for name in ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")


@pytest.fixture
def over_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")


def _capture_login(monkeypatch, provider_module):
    """Record the keyword arguments the CLI passes to a provider's login()."""
    calls: list[dict] = []

    def fake_login(self, **kwargs):
        calls.append(kwargs)
        return {"expires_at": time.time() + 3600}

    monkeypatch.setattr(provider_module, "login", fake_login)
    return calls


@pytest.mark.parametrize(
    "provider,module_path",
    [
        ("anthropic-oauth", "litellm.llms.anthropic.oauth.AnthropicOAuthAuthenticator"),
        ("xai-oauth", "litellm.llms.xai.oauth.XAIOAuthAuthenticator"),
    ],
)
def test_ssh_session_asks_for_a_pasted_code(provider, module_path, isolated_credentials, over_ssh, monkeypatch):
    import importlib

    module_name, _, class_name = module_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    calls = _capture_login(monkeypatch, cls)

    handle_oauth_command((provider, "login"))

    assert calls[0]["prompt_for_code"] is not None, "over SSH the loopback redirect can never arrive"
    assert calls[0]["no_browser"] is True


@pytest.mark.parametrize(
    "provider,module_path",
    [
        ("anthropic-oauth", "litellm.llms.anthropic.oauth.AnthropicOAuthAuthenticator"),
        ("xai-oauth", "litellm.llms.xai.oauth.XAIOAuthAuthenticator"),
    ],
)
def test_local_desktop_uses_the_loopback_flow(provider, module_path, isolated_credentials, local_desktop, monkeypatch):
    import importlib

    module_name, _, class_name = module_path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), class_name)
    calls = _capture_login(monkeypatch, cls)

    handle_oauth_command((provider, "login"))

    assert calls[0]["prompt_for_code"] is None
    assert calls[0]["no_browser"] is False


def test_browser_flag_overrides_remote_detection(isolated_credentials, over_ssh, monkeypatch):
    """`--browser` is for when the loopback port has been forwarded back over SSH."""
    from litellm.llms.xai.oauth import XAIOAuthAuthenticator

    calls = _capture_login(monkeypatch, XAIOAuthAuthenticator)
    handle_oauth_command(("xai-oauth", "login", "--browser"))

    assert calls[0]["prompt_for_code"] is None


def test_manual_flag_forces_the_paste_flow_on_a_desktop(isolated_credentials, local_desktop, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    calls = _capture_login(monkeypatch, AnthropicOAuthAuthenticator)
    handle_oauth_command(("anthropic-oauth", "login", "--manual"))

    assert calls[0]["prompt_for_code"] is not None


def test_force_flag_is_passed_through(isolated_credentials, over_ssh, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    calls = _capture_login(monkeypatch, AnthropicOAuthAuthenticator)
    handle_oauth_command(("anthropic-oauth", "login", "--force"))

    assert calls[0]["force"] is True
