import json
import time

import click
import pytest

from litellm.proxy.oauth_cli import handle_oauth_command


@pytest.fixture(autouse=True)
def deterministic_session(monkeypatch):
    """Pin what the CLI reads from its environment.

    The command chooses between the loopback, prompt and print-and-stop paths from the session and from
    whether stdin is a terminal. Left unpinned, these tests would take different paths depending on where
    the suite itself is run -- under pytest stdin is never a tty, and CI may or may not be over SSH.
    """
    for name in ("SSH_CONNECTION", "SSH_TTY", "SSH_CLIENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr("litellm.proxy.oauth_cli._can_prompt", lambda: True)


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
def local_desktop():
    """The default, pinned by `deterministic_session`."""
    return None


@pytest.fixture
def over_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")


@pytest.fixture
def no_terminal(monkeypatch):
    """A shell that cannot be prompted: a pipe, a script, or a non-interactive remote command."""
    monkeypatch.setattr("litellm.proxy.oauth_cli._can_prompt", lambda: False)


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


# --- a shell that cannot be prompted ------------------------------------------------------------------


@pytest.mark.parametrize("provider,command", [("anthropic-oauth", "anthropic"), ("xai-oauth", "xai")])
def test_without_a_terminal_the_url_is_printed_and_the_login_pauses(
    provider, command, isolated_credentials, over_ssh, no_terminal, monkeypatch, capsys
):
    """`click.prompt` gets EOF in a pipe or a non-interactive shell, so the login is split in two."""
    import importlib

    module = importlib.import_module(f"litellm.llms.{command}.oauth")
    cls = getattr(module, "AnthropicOAuthAuthenticator" if command == "anthropic" else "XAIOAuthAuthenticator")
    monkeypatch.setattr(cls, "begin_login", lambda self: {"authorize_url": "https://example.com/authorize"})

    assert handle_oauth_command((provider, "login")) is True

    out = capsys.readouterr().out
    assert "https://example.com/authorize" in out
    assert f"litellm {provider} login --code" in out, "the second half of the login must be spelled out"


@pytest.mark.parametrize(
    "flags",
    [("login", "--code", "THE-CODE"), ("login", "--code=THE-CODE")],
)
def test_the_code_flag_completes_a_started_login(flags, isolated_credentials, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    supplied: list[str] = []
    monkeypatch.setattr(
        AnthropicOAuthAuthenticator,
        "complete_login",
        lambda self, pasted: supplied.append(pasted) or {"expires_at": time.time() + 3600},
    )

    assert handle_oauth_command(("anthropic-oauth", *flags)) is True
    assert supplied == ["THE-CODE"]


def test_the_code_flag_does_not_start_a_new_login(isolated_credentials, monkeypatch):
    """Starting a new login here would replace the pending verifier the code belongs to."""
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    def fail(self):
        raise AssertionError("begin_login must not run when a code is supplied")

    monkeypatch.setattr(AnthropicOAuthAuthenticator, "begin_login", fail)
    monkeypatch.setattr(
        AnthropicOAuthAuthenticator, "complete_login", lambda self, pasted: {"expires_at": time.time() + 60}
    )

    handle_oauth_command(("anthropic-oauth", "login", "--code", "THE-CODE"))


def test_a_completion_failure_is_reported_cleanly(isolated_credentials, monkeypatch):
    from litellm.llms.anthropic import oauth as anthropic_oauth

    def explode(self, pasted):
        raise anthropic_oauth.AnthropicOAuthError("This code came from a different login attempt.")

    monkeypatch.setattr(anthropic_oauth.AnthropicOAuthAuthenticator, "complete_login", explode)

    with pytest.raises(click.ClickException, match="different login attempt"):
        handle_oauth_command(("anthropic-oauth", "login", "--code", "STALE"))


# --- the instructions match where each provider actually puts the code ---------------------------------


def test_anthropic_is_told_the_code_appears_on_the_page(
    isolated_credentials, over_ssh, no_terminal, monkeypatch, capsys
):
    """Anthropic is asked for `code=true`, so it prints the code rather than redirecting."""
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    monkeypatch.setattr(
        AnthropicOAuthAuthenticator, "begin_login", lambda self: {"authorize_url": "https://claude.ai/x"}
    )
    handle_oauth_command(("anthropic-oauth", "login"))

    out = capsys.readouterr().out
    assert "shows you an authorization code" in out
    assert "#" in out, "the displayed form is code#state, which the user needs to recognise"


def test_xai_is_told_to_read_the_redirect_address(isolated_credentials, over_ssh, no_terminal, monkeypatch, capsys):
    from litellm.llms.xai.oauth import XAIOAuthAuthenticator

    monkeypatch.setattr(XAIOAuthAuthenticator, "begin_login", lambda self: {"authorize_url": "https://x.ai/y"})
    handle_oauth_command(("xai-oauth", "login"))

    out = capsys.readouterr().out
    assert "FAILS TO LOAD" in out
    assert "address bar" in out


@pytest.mark.parametrize("provider", ["anthropic-oauth", "xai-oauth"])
def test_every_provider_warns_against_pasting_the_request_back(
    provider, isolated_credentials, over_ssh, no_terminal, monkeypatch, capsys
):
    import importlib

    module = "anthropic" if provider.startswith("anthropic") else "xai"
    cls_name = "AnthropicOAuthAuthenticator" if module == "anthropic" else "XAIOAuthAuthenticator"
    cls = getattr(importlib.import_module(f"litellm.llms.{module}.oauth"), cls_name)
    monkeypatch.setattr(cls, "begin_login", lambda self: {"authorize_url": "https://example.com/authorize"})

    handle_oauth_command((provider, "login"))
    assert "Do NOT paste the URL above back" in capsys.readouterr().out


# --- through the real command line ---------------------------------------------------------------------
#
# These go through click rather than calling handle_oauth_command directly. Unit tests that skipped click
# passed while every flag was in fact being eaten by the top-level command as an unknown option.


def _invoke(args):
    from click.testing import CliRunner

    from litellm.proxy.proxy_cli import run_server

    return CliRunner().invoke(run_server, args)


def test_the_code_flag_survives_click(isolated_credentials, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    supplied: list[str] = []
    monkeypatch.setattr(
        AnthropicOAuthAuthenticator,
        "complete_login",
        lambda self, pasted: supplied.append(pasted) or {"expires_at": time.time() + 3600},
    )

    result = _invoke(["anthropic-oauth", "login", "--code", "THE-CODE"])

    assert result.exit_code == 0, result.output
    assert supplied == ["THE-CODE"]


@pytest.mark.parametrize("flag", ["--force", "--manual", "--browser", "--no-browser"])
def test_every_login_flag_survives_click(flag, isolated_credentials, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    seen: list[dict] = []
    monkeypatch.setattr(
        AnthropicOAuthAuthenticator,
        "login",
        lambda self, **kwargs: seen.append(kwargs) or {"expires_at": time.time() + 3600},
    )
    monkeypatch.setattr(AnthropicOAuthAuthenticator, "begin_login", lambda self: {"authorize_url": "https://x/y"})

    result = _invoke(["anthropic-oauth", "login", flag])

    assert result.exit_code == 0, result.output
    assert "No such option" not in result.output


@pytest.mark.parametrize("provider", ["anthropic-oauth", "xai-oauth", "chatgpt"])
def test_status_survives_click(provider, isolated_credentials):
    result = _invoke([provider, "status"])

    assert result.exit_code == 0, result.output
    assert "not logged in" in result.output


def test_a_mistyped_proxy_option_is_still_rejected(isolated_credentials):
    """Passing subcommand flags through must not turn typos into silently ignored arguments."""
    result = _invoke(["--porrt", "4000"])

    assert result.exit_code != 0
    assert "No such option" in result.output


def test_proxy_options_before_a_subcommand_still_parse(isolated_credentials, monkeypatch):
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    monkeypatch.setattr(
        AnthropicOAuthAuthenticator, "complete_login", lambda self, pasted: {"expires_at": time.time() + 60}
    )

    result = _invoke(["--host", "0.0.0.0", "anthropic-oauth", "login", "--code", "X"])

    assert result.exit_code == 0, result.output


# --- moving credentials between the file and the database ---------------------------------------------


@pytest.fixture
def fake_database(monkeypatch):
    """Stand in for the database store, so these tests need no database."""
    written = {}

    class _Store:
        def __init__(self, provider, database_url, fallback_path=None):
            self.provider = provider

        def read(self):
            return written.get(self.provider)

        def write(self, data):
            written[self.provider] = dict(data)

        def delete(self):
            return written.pop(self.provider, None) is not None

    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pw@localhost:5432/db")
    monkeypatch.setattr("litellm.proxy.oauth_credential_store.DatabaseCredentialStore", _Store)
    return written


def test_push_copies_the_local_credentials_into_the_database(isolated_credentials, fake_database, capsys):
    _write(
        isolated_credentials,
        "anthropic_oauth_token_dir",
        {"access_token": "sk-ant-oat01-x", "refresh_token": "r", "expires_at": time.time() + 3600},
    )

    assert handle_oauth_command(("anthropic-oauth", "push")) is True

    assert fake_database["anthropic"]["access_token"] == "sk-ant-oat01-x"
    out = capsys.readouterr().out
    assert "copied to database" in out
    assert "sk-ant-oat01-x" not in out


def test_push_leaves_the_local_file_and_says_so(isolated_credentials, fake_database, capsys):
    """Deleting it silently would strand anyone whose database is not yet reachable."""
    path = _write(isolated_credentials, "anthropic_oauth_token_dir", {"access_token": "t", "refresh_token": "r"})

    handle_oauth_command(("anthropic-oauth", "push"))

    assert path.exists()
    assert "left in place" in capsys.readouterr().out


def test_pull_copies_the_database_credentials_to_the_local_file(isolated_credentials, fake_database, capsys):
    fake_database["anthropic"] = {"access_token": "sk-ant-oat01-from-db", "refresh_token": "r"}

    assert handle_oauth_command(("anthropic-oauth", "pull")) is True

    handle_oauth_command(("anthropic-oauth", "status"))
    assert "logged in" in capsys.readouterr().out


def test_pushing_nothing_is_an_error_rather_than_a_silent_success(isolated_credentials, fake_database):
    with pytest.raises(click.ClickException, match="No .* credentials found"):
        handle_oauth_command(("anthropic-oauth", "push"))


def test_pulling_nothing_is_an_error(isolated_credentials, fake_database):
    with pytest.raises(click.ClickException, match="No .* credentials found"):
        handle_oauth_command(("anthropic-oauth", "pull"))


def test_moving_without_a_database_says_which_variable_is_missing(isolated_credentials, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _write(isolated_credentials, "anthropic_oauth_token_dir", {"access_token": "t", "refresh_token": "r"})

    with pytest.raises(click.ClickException, match="DATABASE_URL"):
        handle_oauth_command(("anthropic-oauth", "push"))


@pytest.mark.parametrize(
    "provider,key,env_dir",
    [
        ("anthropic-oauth", "anthropic", "anthropic_oauth_token_dir"),
        ("xai-oauth", "xai", "xai_oauth_token_dir"),
        ("chatgpt", "chatgpt", "chatgpt_token_dir"),
    ],
)
def test_every_provider_can_be_pushed(provider, key, env_dir, isolated_credentials, fake_database):
    _write(isolated_credentials, env_dir, {"access_token": f"token-{key}", "refresh_token": "r"})

    handle_oauth_command((provider, "push"))

    assert fake_database[key]["access_token"] == f"token-{key}"
