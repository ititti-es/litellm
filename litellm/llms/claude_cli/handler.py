"""A LiteLLM provider that spends a Claude subscription by running Claude Code.

Registered through `litellm.custom_provider_map`, so nothing in the core dispatch has to know about it:

    litellm_settings:
      custom_provider_map:
        - provider: claude_cli
          custom_handler: litellm.llms.claude_cli.handler.claude_cli_handler

    model_list:
      - model_name: claude-sub
        litellm_params:
          model: claude_cli/claude-sonnet-5

Each request runs `claude --print`. A conversation is continued with `--resume`, so Claude Code keeps its
own prefix warm and only the newest message is sent; the session id is derived from the caller's session,
so no mapping is stored and a restart resumes the same conversation.
"""

import asyncio
import os
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol

from litellm._logging import verbose_logger
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse

from .common_utils import (
    CLAUDE_CLI_DEFAULT_TIMEOUT,
    CLAUDE_CLI_NO_TOOLS,
    CLAUDE_CLI_TOKEN_FD,
    CLAUDE_CLI_TOKEN_FD_ENV,
    ClaudeCLIError,
    binary_path,
    is_missing_session,
    session_uuid,
)
from .sync import SyncRequest, sync_transcript
from .transformation import CliResult, parse_cli_output, render_prompt, system_prompt_of, to_model_response


class CommandRunner(Protocol):
    """Runs the CLI and returns (returncode, stdout, stderr)."""

    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float
    ) -> tuple[int, str, str]: ...


class AsyncCommandRunner(Protocol):
    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float
    ) -> Awaitable[tuple[int, str, str]]: ...


_SCRUBBED_ENV: Final = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB")
_STARTED_SESSIONS: Final[set[str]] = set()  # mutable-ok: process-local record of sessions already opened
_RESUME_FLAG: Final = "--resume"
_NEW_SESSION_FLAG: Final = "--session-id"
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


def _params(litellm_params: Mapping[str, object] | None) -> Mapping[str, object]:
    return litellm_params or _NO_PARAMS


def _access_token(params: Mapping[str, object], api_key: str | None) -> str:
    """Resolve the credential Claude Code should run as."""
    explicit: Final = api_key or params.get("api_key")
    if isinstance(explicit, str) and explicit:
        return explicit
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator, AnthropicOAuthError

    try:
        return AnthropicOAuthAuthenticator().get_access_token()
    except AnthropicOAuthError as exc:
        raise ClaudeCLIError(str(exc), status_code=401) from exc


def _session_key(params: Mapping[str, object]) -> str | None:
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    metadata: Final = params.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("litellm_session_id", "session_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _owner(params: Mapping[str, object]) -> str:
    metadata: Final = params.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("user_api_key_hash", "user_api_key"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value
    return "anonymous"


def _allowed_tools(params: Mapping[str, object]) -> str | None:
    """Tools a deployment has explicitly opted into, if any.

    Claude Code can read files and run commands. Serving requests with that enabled gives every caller code
    execution on the proxy host, so it stays off unless a deployment names what it wants.
    """
    value: Final = params.get("claude_cli_allowed_tools")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, (list, tuple)) and value:
        return ",".join(str(item) for item in value)
    return None


def _timeout_of(params: Mapping[str, object], timeout: float | None) -> float:
    if isinstance(timeout, (int, float)) and timeout > 0:
        return float(timeout)
    configured: Final = params.get("claude_cli_timeout")
    if isinstance(configured, (int, float)) and configured > 0:
        return float(configured)
    return CLAUDE_CLI_DEFAULT_TIMEOUT


def _child_env(params: Mapping[str, object]) -> Mapping[str, str]:
    env = dict(os.environ)  # mutable-ok: the child's environment, built per call
    for name in _SCRUBBED_ENV:
        env.pop(name, None)
    env[CLAUDE_CLI_TOKEN_FD_ENV] = str(CLAUDE_CLI_TOKEN_FD)
    config_dir: Final = params.get("claude_cli_config_dir")
    if isinstance(config_dir, str) and config_dir:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(config_dir)
    return env


def _build_command(
    model: str,
    prompt: str,
    system: str,
    session: str | None,
    resume: bool,
    params: Mapping[str, object],
) -> tuple[str, ...]:
    command = (
        binary_path(),
        "--print",
        prompt,
        "--output-format",
        "json",
        "--system-prompt",
        system,
    )
    if model:
        command += ("--model", model)
    allowed: Final = _allowed_tools(params)
    if allowed is None:
        command += ("--disallowed-tools", CLAUDE_CLI_NO_TOOLS)
    else:
        command += ("--allowed-tools", allowed)
    if session:
        command += (_RESUME_FLAG if resume else _NEW_SESSION_FLAG, session)
    return command


class _TokenPipe:
    """Hands the credential to the child on a file descriptor rather than in its environment."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.read_fd = -1
        self.write_fd = -1

    def __enter__(self) -> "_TokenPipe":
        self.read_fd, self.write_fd = os.pipe()
        os.set_inheritable(self.read_fd, True)
        return self

    def deliver(self) -> None:
        os.write(self.write_fd, self.token.encode())
        os.close(self.write_fd)
        self.write_fd = -1

    def close_parent_read(self) -> None:
        if self.read_fd != -1:
            os.close(self.read_fd)
            self.read_fd = -1

    def __exit__(self, *_: object) -> None:
        for fd in (self.read_fd, self.write_fd):
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.read_fd = self.write_fd = -1


def _run(command: Sequence[str], env: Mapping[str, str], token: str, timeout: float) -> tuple[int, str, str]:
    with _TokenPipe(token) as pipe:
        try:
            # stdin is closed explicitly: Claude Code otherwise waits several seconds for piped input.
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),  # mutable-ok: the child's environment, required as a dict
                pass_fds=(pipe.read_fd,),
                close_fds=True,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.",
                status_code=500,
            ) from exc
        pipe.close_parent_read()
        pipe.deliver()
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            proc.communicate()
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504) from exc
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _run_async(
    command: Sequence[str], env: Mapping[str, str], token: str, timeout: float
) -> tuple[int, str, str]:
    with _TokenPipe(token) as pipe:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=dict(env),  # mutable-ok: the child's environment, required as a dict
                pass_fds=(pipe.read_fd,),
                close_fds=True,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.",
                status_code=500,
            ) from exc
        pipe.close_parent_read()
        pipe.deliver()
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            proc.kill()
            await proc.communicate()
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504) from exc
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _plan(
    model: str,
    messages: Sequence[Mapping[str, object]],
    litellm_params: Mapping[str, object] | None,
    api_key: str | None,
    timeout: float | None,
) -> tuple[str, str, str | None, bool, Mapping[str, str], str, float]:
    """Work out the call: model, prompt, session, whether to resume, environment, token and timeout."""
    params: Final = _params(litellm_params)
    provider_model: Final = model.split("/", 1)[-1] if "/" in model else model
    session_key: Final = _session_key(params)
    session: Final = session_uuid(_owner(params), session_key) if session_key else None
    resume: Final = bool(session and session in _STARTED_SESSIONS)
    return (
        provider_model,
        render_prompt(messages, latest_only=resume),
        session,
        resume,
        _child_env(params),
        _access_token(params, api_key),
        _timeout_of(params, timeout),
    )


def _projects_dir(params: Mapping[str, object]) -> str | None:
    config_dir: Final = params.get("claude_cli_config_dir")
    return (
        os.path.join(os.path.expanduser(config_dir), "projects") if isinstance(config_dir, str) and config_dir else None
    )


def _record_transcript(result: CliResult, session: str | None, params: Mapping[str, object]) -> None:
    """Hand the finished turn to the transcript store, which reads it from Claude Code's own record.

    The CLI reports the session it actually used, which is the one we asked for on a resumed conversation and
    one of its own choosing on a single-shot call; either way it is what the transcript is filed under.
    """
    session_id: Final = result.get("session_id") or session
    if not session_id:
        return
    sync_transcript(
        SyncRequest(
            session_id=session_id,
            owner=_owner(params),
            client_session_id=_session_key(params),
            projects_dir=_projects_dir(params),
        )
    )


def _finish(
    returncode: int,
    stdout: str,
    stderr: str,
    session: str | None,
    model: str,
    model_response: ModelResponse,
    params: Mapping[str, object] = _NO_PARAMS,
) -> ModelResponse:
    if returncode != 0:
        raise ClaudeCLIError(f"Claude CLI exited {returncode}: {(stderr or stdout).strip()[:400]}")
    result: Final = parse_cli_output(stdout)
    if session:
        _STARTED_SESSIONS.add(session)
    _record_transcript(result, session, params)
    return to_model_response(result, model, model_response)


class ClaudeCLIHandler(CustomLLM):
    """Runs Claude Code for each request, resuming its session between turns.

    The two runners are injected so tests can drive the whole path -- argv, session handling, the retry on a
    vanished session -- without a real binary or a real subscription.
    """

    def __init__(self, run: CommandRunner | None = None, run_async: AsyncCommandRunner | None = None) -> None:
        super().__init__()
        self._run = run or _run
        self._run_async = run_async or _run_async

    def completion(  # pyright: ignore[reportIncompatibleMethodOverride]  # base declares a broader signature than this provider needs
        self,
        model: str,
        messages: list,  # mutable-ok: matches CustomLLM's declared signature
        api_base: str = "",
        custom_prompt_dict: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        model_response: ModelResponse | None = None,
        print_verbose: Callable | None = None,
        encoding: object = None,
        api_key: str | None = None,
        logging_obj: object = None,
        optional_params: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        acompletion: object = None,
        litellm_params: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        logger_fn: object = None,
        headers: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        timeout: float | None = None,
        client: object = None,
    ) -> ModelResponse:
        provider_model, prompt, session, resume, env, token, wait = _plan(
            model, messages, litellm_params, api_key, timeout
        )
        system: Final = system_prompt_of(messages)
        params: Final = _params(litellm_params)
        command = _build_command(provider_model, prompt, system, session, resume, params)
        returncode, stdout, stderr = self._run(command, env, token, wait)

        if returncode != 0 and resume and is_missing_session(stderr):
            # The session was recorded as started but Claude Code no longer has it -- its history may have
            # been cleared. Start it again, this time sending the whole conversation.
            verbose_logger.info("Claude CLI session %s is gone; starting a new one", session)
            _STARTED_SESSIONS.discard(session or "")
            command = _build_command(
                provider_model, render_prompt(messages, latest_only=False), system, session, False, params
            )
            returncode, stdout, stderr = self._run(command, env, token, wait)

        return _finish(returncode, stdout, stderr, session, provider_model, model_response or ModelResponse(), params)

    async def acompletion(  # pyright: ignore[reportIncompatibleMethodOverride]  # see completion
        self,
        model: str,
        messages: list,  # mutable-ok: matches CustomLLM's declared signature
        api_base: str = "",
        custom_prompt_dict: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        model_response: ModelResponse | None = None,
        print_verbose: Callable | None = None,
        encoding: object = None,
        api_key: str | None = None,
        logging_obj: object = None,
        optional_params: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        acompletion: object = None,
        litellm_params: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        logger_fn: object = None,
        headers: dict | None = None,  # mutable-ok: matches CustomLLM's declared signature
        timeout: float | None = None,
        client: object = None,
    ) -> ModelResponse:
        provider_model, prompt, session, resume, env, token, wait = _plan(
            model, messages, litellm_params, api_key, timeout
        )
        system: Final = system_prompt_of(messages)
        params: Final = _params(litellm_params)
        command = _build_command(provider_model, prompt, system, session, resume, params)
        returncode, stdout, stderr = await self._run_async(command, env, token, wait)

        if returncode != 0 and resume and is_missing_session(stderr):
            verbose_logger.info("Claude CLI session %s is gone; starting a new one", session)
            _STARTED_SESSIONS.discard(session or "")
            command = _build_command(
                provider_model, render_prompt(messages, latest_only=False), system, session, False, params
            )
            returncode, stdout, stderr = await self._run_async(command, env, token, wait)

        return _finish(returncode, stdout, stderr, session, provider_model, model_response or ModelResponse(), params)


claude_cli_handler: Final = ClaudeCLIHandler()
