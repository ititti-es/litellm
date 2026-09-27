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
import signal
import subprocess
import tempfile
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import IO, Final, NamedTuple, Protocol, TypeAlias

from litellm._logging import verbose_logger
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse

from .common_utils import (
    CLAUDE_CLI_DEFAULT_TIMEOUT,
    CLAUDE_CLI_NO_TOOLS,
    CLAUDE_CLI_TOKEN_FD_ENV,
    ClaudeCLIError,
    binary_path,
    is_missing_session,
    is_session_in_use,
    session_uuid,
)
from .sync import SyncRequest, sync_transcript
from .transformation import (
    CliResult,
    final_chunk,
    parse_cli_output,
    render_prompt,
    stream_result,
    stream_text,
    system_prompt_of,
    text_chunk,
    to_model_response,
)


class CommandRunner(Protocol):
    """Runs the CLI and returns (returncode, stdout, stderr)."""

    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
    ) -> tuple[int, str, str]: ...


class AsyncCommandRunner(Protocol):
    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
    ) -> Awaitable[tuple[int, str, str]]: ...


class StreamExit(NamedTuple):
    """The last item a stream runner yields, once the process has ended."""

    returncode: int
    stderr: str


StreamItem: TypeAlias = str | StreamExit


class StreamRunner(Protocol):
    """Runs the CLI, yielding each stdout line as it arrives and then a StreamExit."""

    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
    ) -> Iterator[StreamItem]: ...


class AsyncStreamRunner(Protocol):
    def __call__(
        self, command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
    ) -> AsyncIterator[StreamItem]: ...


_SCRUBBED_ENV: Final = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB")
_STARTED_SESSIONS: Final[set[str]] = set()  # mutable-ok: process-local record of sessions already opened
_RESUME_FLAG: Final = "--resume"
_NEW_SESSION_FLAG: Final = "--session-id"
_NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})
_JSON_FORMAT: Final = ("--output-format", "json")
# Partial messages are what make this a stream of tokens rather than of whole turns; stream-json needs --verbose.
_STREAM_FORMAT: Final = ("--output-format", "stream-json", "--verbose", "--include-partial-messages")
# One stream-json line can carry a whole assistant message, well past asyncio's 64 KiB readline default.
_STREAM_LINE_LIMIT: Final = 64 * 1024 * 1024


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
    config_dir: Final = params.get("claude_cli_config_dir")
    if isinstance(config_dir, str) and config_dir:
        env["CLAUDE_CONFIG_DIR"] = os.path.expanduser(config_dir)
    return env


def _build_command(
    model: str,
    system: str,
    session: str | None,
    resume: bool,
    params: Mapping[str, object],
    streaming: bool = False,
) -> tuple[str, ...]:
    command = (
        binary_path(),
        "--print",
        *(_STREAM_FORMAT if streaming else _JSON_FORMAT),
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

    def child_env(self, env: Mapping[str, str]) -> dict[str, str]:
        """The child's environment, naming the descriptor the token will actually arrive on.

        `pass_fds` keeps a descriptor's number in the child, and `os.pipe()` hands out whichever number is free
        -- almost never 3 in a long-running server. A fixed number here would point Claude Code at a closed
        descriptor, and it would silently fall back to whatever login it finds on its own.
        """
        return {**env, CLAUDE_CLI_TOKEN_FD_ENV: str(self.read_fd)}  # mutable-ok: the child's environment

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


def _end(proc: "subprocess.Popen[bytes] | asyncio.subprocess.Process") -> None:
    """Kill the run and everything it started.

    Each run is its own process group (`start_new_session`), so this reaches children Claude Code spawned
    itself; killing only the leader would orphan them.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run(
    command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
) -> tuple[int, str, str]:
    with _TokenPipe(token) as pipe:
        try:
            # The prompt goes on stdin: one argv string is capped at 128 KiB and readable by any local user.
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=pipe.child_env(env),
                pass_fds=(pipe.read_fd,),
                close_fds=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.",
                status_code=500,
            ) from exc
        pipe.close_parent_read()
        pipe.deliver()
        try:
            stdout, stderr = proc.communicate(input=prompt.encode(), timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _end(proc)
            proc.communicate()
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504) from exc
    return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _run_async(
    command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
) -> tuple[int, str, str]:
    with _TokenPipe(token) as pipe:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=pipe.child_env(env),
                pass_fds=(pipe.read_fd,),
                close_fds=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.",
                status_code=500,
            ) from exc
        pipe.close_parent_read()
        pipe.deliver()
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=prompt.encode()), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            _end(proc)
            await proc.communicate()
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504) from exc
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


def _feed(stdin: IO[bytes] | None, prompt: str) -> None:
    if stdin is None:
        return
    try:
        stdin.write(prompt.encode())
        stdin.close()
    except BrokenPipeError:
        pass  # the child exited first; its exit status says why


async def _feed_async(stdin: asyncio.StreamWriter | None, prompt: str) -> None:
    if stdin is None:
        return
    try:
        stdin.write(prompt.encode())
        await stdin.drain()
        stdin.close()
    except (BrokenPipeError, ConnectionResetError):
        pass  # the child exited first; its exit status says why


def _not_found(command: Sequence[str], exc: FileNotFoundError) -> ClaudeCLIError:
    return ClaudeCLIError(
        f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.", status_code=500
    )


def _stream(
    command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
) -> Iterator[StreamItem]:
    """Yield stdout a line at a time, then how the process ended.

    stderr goes to a temporary file rather than a pipe: nothing reads a pipe while stdout is being consumed, so
    a chatty stderr could otherwise fill it and stall the child. Closing the generator early -- a caller who
    disconnects mid-reply -- kills the process rather than leaving it running to no one.
    """
    timed_out: Final = threading.Event()
    with _TokenPipe(token) as pipe, tempfile.TemporaryFile() as errors:
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                env=pipe.child_env(env),
                pass_fds=(pipe.read_fd,),
                close_fds=True,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise _not_found(command, exc) from exc
        pipe.close_parent_read()
        pipe.deliver()

        def expire() -> None:
            timed_out.set()
            _end(proc)

        watchdog: Final = threading.Timer(timeout, expire)
        watchdog.start()
        try:
            _feed(proc.stdin, prompt)
            for raw in proc.stdout or ():
                yield raw.decode(errors="replace")
            proc.wait()
        finally:
            watchdog.cancel()
            if proc.poll() is None:
                _end(proc)
                proc.wait()
        if timed_out.is_set():
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504)
        errors.seek(0)
        yield StreamExit(proc.returncode, errors.read().decode(errors="replace"))


async def _stream_async(
    command: Sequence[str], env: Mapping[str, str], token: str, timeout: float, prompt: str = ""
) -> AsyncIterator[StreamItem]:
    """The async counterpart of `_stream`; stderr is drained by a task so it can never stall the child."""
    with _TokenPipe(token) as pipe:
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=pipe.child_env(env),
                pass_fds=(pipe.read_fd,),
                close_fds=True,
                start_new_session=True,
                limit=_STREAM_LINE_LIMIT,
            )
        except FileNotFoundError as exc:
            raise _not_found(command, exc) from exc
        pipe.close_parent_read()
        pipe.deliver()
        errors: Final = asyncio.ensure_future(proc.stderr.read() if proc.stderr else asyncio.sleep(0, b""))
        try:
            async with asyncio.timeout(timeout):
                await _feed_async(proc.stdin, prompt)
                while line := await proc.stdout.readline() if proc.stdout else b"":
                    yield line.decode(errors="replace")
                returncode: Final = await proc.wait()
        except TimeoutError as exc:
            raise ClaudeCLIError(f"Claude CLI timed out after {timeout:g}s", status_code=504) from exc
        finally:
            if proc.returncode is None:
                _end(proc)
                await proc.wait()
        yield StreamExit(returncode, (await errors).decode(errors="replace"))


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


@dataclass(slots=True)
class _StreamedTurn:
    """What one streamed run produced, gathered as its lines go past."""

    spoke: bool = False
    result: CliResult | None = None
    exit: StreamExit | None = None

    def take(self, item: StreamItem) -> GenericStreamingChunk | None:
        """The chunk to hand the caller for this item, if it carries any text."""
        if isinstance(item, StreamExit):
            self.exit = item
            return None
        text: Final = stream_text(item)
        if text is not None:
            self.spoke = True
            return text_chunk(text)
        result: Final = stream_result(item)
        if result is not None:
            self.result = result
        return None

    def session_unusable(self, resumed: bool) -> bool:
        """Whether the planned session could not be used, failed before saying anything.

        Only then is a retry safe: once text has reached the caller, a second run would repeat it.
        """
        return (
            not self.spoke
            and self.exit is not None
            and self.exit.returncode != 0
            and _session_unusable(resumed, self.exit.stderr)
        )

    def closing(self, session: str | None, params: Mapping[str, object]) -> GenericStreamingChunk:
        ended: Final = self.exit
        if ended is None or ended.returncode != 0 or self.result is None:
            code: Final = ended.returncode if ended is not None else "?"
            detail: Final = (ended.stderr if ended is not None else "").strip()[:400] or "no result record"
            raise ClaudeCLIError(f"Claude CLI exited {code}: {detail}")
        chunk: Final = final_chunk(self.result)
        if session:
            _STARTED_SESSIONS.add(session)
        _record_transcript(self.result, session, params)
        return chunk


def _session_unusable(resumed: bool, stderr: str) -> bool:
    """A resume found no session, or a fresh start found one already there (started before a restart, or by
    another worker)"""
    return is_missing_session(stderr) if resumed else is_session_in_use(stderr)


def _attempts(
    model: str,
    messages: Sequence[Mapping[str, object]],
    prompt: str,
    session: str | None,
    resume: bool,
    params: Mapping[str, object],
) -> tuple[tuple[tuple[str, ...], str, bool], ...]:
    """The runs to try, in order: as planned, then with the session taken the other way round."""
    system: Final = system_prompt_of(messages)
    planned: Final = (_build_command(model, system, session, resume, params, streaming=True), prompt, resume)
    if not session:
        return (planned,)
    other: Final = render_prompt(messages, latest_only=not resume)
    return (planned, (_build_command(model, system, session, not resume, params, streaming=True), other, not resume))


class ClaudeCLIHandler(CustomLLM):
    """Runs Claude Code for each request, resuming its session between turns.

    The two runners are injected so tests can drive the whole path -- argv, session handling, the retry on a
    vanished session -- without a real binary or a real subscription.
    """

    def __init__(
        self,
        run: CommandRunner | None = None,
        run_async: AsyncCommandRunner | None = None,
        stream: StreamRunner | None = None,
        stream_async: AsyncStreamRunner | None = None,
    ) -> None:
        super().__init__()
        self._run = run or _run
        self._run_async = run_async or _run_async
        self._stream = stream or _stream
        self._stream_async = stream_async or _stream_async

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
        command = _build_command(provider_model, system, session, resume, params)
        returncode, stdout, stderr = self._run(command, env, token, wait, prompt=prompt)

        if returncode != 0 and session and _session_unusable(resume, stderr):
            verbose_logger.info(
                "Claude CLI session %s cannot be %s; retrying", session, "resumed" if resume else "started"
            )
            _STARTED_SESSIONS.discard(session)
            command = _build_command(provider_model, system, session, not resume, params)
            returncode, stdout, stderr = self._run(
                command, env, token, wait, prompt=render_prompt(messages, latest_only=not resume)
            )

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
        command = _build_command(provider_model, system, session, resume, params)
        returncode, stdout, stderr = await self._run_async(command, env, token, wait, prompt=prompt)

        if returncode != 0 and session and _session_unusable(resume, stderr):
            verbose_logger.info(
                "Claude CLI session %s cannot be %s; retrying", session, "resumed" if resume else "started"
            )
            _STARTED_SESSIONS.discard(session)
            command = _build_command(provider_model, system, session, not resume, params)
            returncode, stdout, stderr = await self._run_async(
                command, env, token, wait, prompt=render_prompt(messages, latest_only=not resume)
            )

        return _finish(returncode, stdout, stderr, session, provider_model, model_response or ModelResponse(), params)

    def streaming(  # pyright: ignore[reportIncompatibleMethodOverride]  # see completion
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
    ) -> Iterator[GenericStreamingChunk]:
        """Stream the reply as Claude Code produces it; every agent harness asks for this."""
        provider_model, prompt, session, resume, env, token, wait = _plan(
            model, messages, litellm_params, api_key, timeout
        )
        params: Final = _params(litellm_params)
        turn: _StreamedTurn = _StreamedTurn()  # rebind-ok: one per attempt
        for command, attempt_prompt, resumed in _attempts(provider_model, messages, prompt, session, resume, params):
            turn = _StreamedTurn()
            for item in self._stream(command, env, token, wait, prompt=attempt_prompt):
                if (chunk := turn.take(item)) is not None:
                    yield chunk
            if not turn.session_unusable(resumed):
                break
            verbose_logger.info(
                "Claude CLI session %s cannot be %s; retrying", session, "resumed" if resumed else "started"
            )
            _STARTED_SESSIONS.discard(session or "")
        yield turn.closing(session, params)

    async def astreaming(  # pyright: ignore[reportIncompatibleMethodOverride]  # see completion
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
    ) -> AsyncIterator[GenericStreamingChunk]:
        provider_model, prompt, session, resume, env, token, wait = _plan(
            model, messages, litellm_params, api_key, timeout
        )
        params: Final = _params(litellm_params)
        turn: _StreamedTurn = _StreamedTurn()  # rebind-ok: one per attempt
        for command, attempt_prompt, resumed in _attempts(provider_model, messages, prompt, session, resume, params):
            turn = _StreamedTurn()
            async for item in self._stream_async(command, env, token, wait, prompt=attempt_prompt):
                if (chunk := turn.take(item)) is not None:
                    yield chunk
            if not turn.session_unusable(resumed):
                break
            verbose_logger.info(
                "Claude CLI session %s cannot be %s; retrying", session, "resumed" if resumed else "started"
            )
            _STARTED_SESSIONS.discard(session or "")
        yield turn.closing(session, params)


claude_cli_handler: Final = ClaudeCLIHandler()
