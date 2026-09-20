"""`litellm <provider> login|logout|status` for subscription-backed providers.

A gateway must not discover that it needs a login in the middle of serving a request: the ChatGPT device
flow, for instance, prints a code and blocks for up to fifteen minutes. These commands move every login to
an explicit, interactive step an operator runs before the proxy starts.

The provider modules own the flows; this only dispatches to them and reports what happened. Stored records
are read as plain JSON rather than through each authenticator, so this stays clear of their internals.
"""

import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import click

from litellm.llms.base_llm.oauth import is_remote_session

ACTIONS: Final = ("login", "logout", "status")


class _OAuthCommandError(Exception):
    """A login command failed for a reason worth printing without a traceback."""


def _read_record(path: str) -> Mapping[str, object] | None:
    try:
        with open(path, "r") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _delete_record(path: str) -> bool:
    try:
        os.unlink(path)
        return True
    except OSError:
        return False


def _fmt_expiry(expires_at: object) -> str:
    if not isinstance(expires_at, (int, float)):
        return "unknown"
    remaining: Final = float(expires_at) - time.time()
    stamp: Final = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(expires_at)))
    if remaining <= 0:
        return f"{stamp} (expired)"
    return f"{stamp} (in {int(remaining // 60)}m)"


def _report_login(name: str, auth_file: str, expires_at: object) -> None:
    click.echo(f"{name} OAuth login successful. Credentials saved to {auth_file}.")
    if expires_at is not None:
        click.echo(f"Access token expires at {_fmt_expiry(expires_at)}.")


def _report_status(name: str, auth_file: str, login_command: str) -> None:
    record: Final = _read_record(auth_file)
    if not record or not record.get("access_token"):
        click.echo(f"{name}: not logged in. Run `{login_command}`.")
        return
    click.echo(f"{name}: logged in ({auth_file}).")
    click.echo(f"  access token expires at {_fmt_expiry(record.get('expires_at'))}")
    for label, key in (("scope", "scope"), ("account", "account_id")):
        value = record.get(key)
        if isinstance(value, str) and value:
            click.echo(f"  {label}: {value}")


def _report_logout(name: str, removed: bool) -> None:
    click.echo(f"{name}: credentials removed." if removed else f"{name}: no stored credentials.")


def _flag_value(flags: Sequence[str], name: str) -> str | None:
    """Read `--name value` or `--name=value` out of the passthrough arguments."""
    for index, flag in enumerate(flags):
        if flag == name and index + 1 < len(flags):
            return flags[index + 1]
        if flag.startswith(f"{name}="):
            return flag.split("=", 1)[1]
    return None


def _explain_url(authorize_url: str) -> None:
    click.echo("1. Open this URL in a browser on any machine:")
    click.echo("")
    click.echo(f"   {authorize_url}")
    click.echo("")
    click.echo("2. Approve the request there.")
    click.echo("3. The browser then lands on a 127.0.0.1 page that FAILS TO LOAD. That is the success case.")
    click.echo("4. Copy that failed page's address (not the URL above) -- the code is in it.")


def _prompt_for_code(authorize_url: str) -> str:
    _explain_url(authorize_url)
    return click.prompt("Authorization code (or the full redirect URL)", type=str)


def _can_prompt() -> bool:
    """Whether there is a terminal to read an answer from.

    Run from a script, a pipe or a non-interactive shell, `click.prompt` gets EOF and aborts at once. The
    login is then split in two: print the URL now, take the code in a second command.
    """
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _wants_paste(flags: Sequence[str]) -> bool:
    """Whether to ask the user for the code instead of waiting on a loopback redirect.

    A remote session is the default case for this: the redirect would arrive at the machine running the
    browser, not this one, so the wait could only ever time out. `--browser` forces the loopback flow back
    on, for when the port is forwarded.
    """
    if "--browser" in flags:
        return False
    return "--manual" in flags or is_remote_session()


def _start_login(authorize_url: str, command: str) -> None:
    """Print the URL and how to finish, for a shell that cannot be prompted."""
    _explain_url(authorize_url)
    click.echo("")
    click.echo("Then finish the login with:")
    click.echo(f"  {command} --code '<the code>'")
    click.echo("")
    click.echo("The link is valid for 15 minutes.")


def _anthropic(action: str, flags: Sequence[str]) -> None:
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator, AnthropicOAuthError

    authenticator: Final = AnthropicOAuthAuthenticator()
    name: Final = "Anthropic (Claude Pro/Max)"
    if action == "status":
        _report_status(name, authenticator.auth_file, "litellm anthropic-oauth login")
        return
    if action == "logout":
        _report_logout(name, authenticator.logout())
        return
    supplied: Final = _flag_value(flags, "--code")
    try:
        if supplied is not None:
            record = authenticator.complete_login(supplied)
        else:
            paste = _wants_paste(flags)
            if paste and not _can_prompt():
                _start_login(authenticator.begin_login()["authorize_url"], "litellm anthropic-oauth login")
                return
            record = authenticator.login(
                force="--force" in flags,
                no_browser=paste or "--no-browser" in flags,
                prompt_for_code=_prompt_for_code if paste else None,
            )
    except AnthropicOAuthError as exc:
        raise _OAuthCommandError(str(exc)) from exc
    _report_login(name, authenticator.auth_file, record.get("expires_at"))


def _xai(action: str, flags: Sequence[str]) -> None:
    from litellm.llms.xai.oauth import XAIOAuthAuthenticator, XAIOAuthError

    authenticator: Final = XAIOAuthAuthenticator()
    name: Final = "xAI (SuperGrok)"
    if action == "status":
        _report_status(name, authenticator.auth_file, "litellm xai-oauth login")
        return
    if action == "logout":
        _report_logout(name, _delete_record(authenticator.auth_file))
        return
    supplied: Final = _flag_value(flags, "--code")
    try:
        if supplied is not None:
            record = authenticator.complete_login(supplied)
        else:
            paste = _wants_paste(flags)
            if paste and not _can_prompt():
                _start_login(authenticator.begin_login()["authorize_url"], "litellm xai-oauth login")
                return
            record = authenticator.login(
                force="--force" in flags,
                no_browser=paste or "--no-browser" in flags,
                prompt_for_code=_prompt_for_code if paste else None,
            )
    except XAIOAuthError as exc:
        raise _OAuthCommandError(str(exc)) from exc
    _report_login(name, authenticator.auth_file, record.get("expires_at"))


def _chatgpt(action: str, flags: Sequence[str]) -> None:
    from litellm.llms.chatgpt.authenticator import Authenticator
    from litellm.llms.chatgpt.common_utils import GetAccessTokenError, GetDeviceCodeError

    authenticator: Final = Authenticator()
    name: Final = "ChatGPT (Codex subscription)"
    if action == "status":
        _report_status(name, authenticator.auth_file, "litellm chatgpt login")
        return
    if action == "logout":
        _report_logout(name, _delete_record(authenticator.auth_file))
        return
    if "--force" in flags:
        _delete_record(authenticator.auth_file)
    try:
        authenticator.get_access_token()
    except (GetAccessTokenError, GetDeviceCodeError) as exc:
        raise _OAuthCommandError(str(exc)) from exc
    stored: Final = _read_record(authenticator.auth_file)
    _report_login(name, authenticator.auth_file, stored.get("expires_at") if stored else None)


_PROVIDERS: Final[Mapping[str, Callable[[str, Sequence[str]], None]]] = {  # mutable-ok: dispatch table constant
    "anthropic-oauth": _anthropic,
    "xai-oauth": _xai,
    "chatgpt": _chatgpt,
}


def handle_oauth_command(cli_args: Sequence[str]) -> bool:
    """Run `<provider> <action> [flags]` if that is what was asked for.

    Returns True when the arguments were handled, so the caller can fall through to starting the proxy for
    anything else.
    """
    if len(cli_args) < 2:
        return False
    provider, action = cli_args[0], cli_args[1]
    runner: Final = _PROVIDERS.get(provider)
    if runner is None:
        return False
    if action not in ACTIONS:
        raise click.UsageError(f"Unknown action '{action}' for {provider}. Expected one of: {', '.join(ACTIONS)}.")
    try:
        runner(action, tuple(cli_args[2:]))
    except _OAuthCommandError as exc:
        raise click.ClickException(str(exc)) from exc
    return True
