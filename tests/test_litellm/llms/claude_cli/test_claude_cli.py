import json
import os

import pytest

from litellm.llms.claude_cli.common_utils import (
    CLAUDE_CLI_DEFAULT_SYSTEM_PROMPT,
    ClaudeCLIError,
    is_missing_session,
    session_uuid,
)
from litellm.llms.claude_cli.handler import ClaudeCLIHandler
from litellm.llms.claude_cli.transformation import (
    stream_text,
    parse_cli_output,
    render_prompt,
    system_prompt_of,
    to_model_response,
    usage_of,
)
from litellm.types.utils import ModelResponse


@pytest.fixture(autouse=True)
def _no_real_login(monkeypatch):
    """Tests must not read the machine's own Claude login; before this one they silently depended on it."""
    from litellm.llms.anthropic.oauth import AnthropicOAuthAuthenticator

    monkeypatch.setattr(AnthropicOAuthAuthenticator, "get_access_token", lambda self: "sk-ant-oat01-fixture")


def _cli_json(result="ok", **overrides):
    payload = {
        "result": result,
        "session_id": "4f0af742-dddb-4b02-95f1-ae56834c6cb2",
        "is_error": False,
        "subtype": "success",
        "stop_reason": "end_turn",
        "total_cost_usd": 0.002,
        "usage": {
            "input_tokens": 4,
            "output_tokens": 8,
            "cache_read_input_tokens": 2000,
            "cache_creation_input_tokens": 100,
        },
    }
    payload.update(overrides)
    return json.dumps(payload)


class _Runner:
    """Stands in for the subprocess, recording each argv it was asked to run."""

    def __init__(self, *results):
        self.results = list(results) or [(0, _cli_json(), "")]
        self.calls = []

    def __call__(self, command, env, token, timeout, prompt=""):
        self.calls.append(
            {"command": list(command), "env": dict(env), "token": token, "timeout": timeout, "prompt": prompt}
        )
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def flag(self, name, call=0):
        """Return the value following a flag in a recorded argv, or None."""
        argv = self.calls[call]["command"]
        return argv[argv.index(name) + 1] if name in argv else None


def _messages(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]


def _complete(handler, messages, **params):
    return handler.completion(
        model="claude_cli/claude-sonnet-5",
        messages=messages,
        model_response=ModelResponse(),
        litellm_params=params,
    )


# --- session identity ------------------------------------------------------------------------------


def test_the_session_id_is_derived_not_stored():
    """Deriving it means a restart resumes the same conversation with no mapping to keep."""
    assert session_uuid("owner", "sess-1") == session_uuid("owner", "sess-1")


def test_two_owners_never_share_a_session():
    """The caller chooses the session key, so it must not be able to reach another key's conversation."""
    assert session_uuid("owner-a", "sess-1") != session_uuid("owner-b", "sess-1")


def test_the_session_id_is_a_uuid():
    import uuid

    assert uuid.UUID(session_uuid("owner", "sess-1")).version == 5


# --- what gets run ---------------------------------------------------------------------------------


def test_the_first_turn_opens_a_chosen_session():
    runner = _Runner()
    handler = ClaudeCLIHandler(run=runner)

    _complete(handler, _messages(("user", "hello")), litellm_session_id="s-first")

    assert runner.flag("--session-id") == session_uuid("anonymous", "s-first")
    assert "--resume" not in runner.calls[0]["command"]


def test_a_later_turn_resumes_and_sends_only_the_new_message():
    """Claude Code already holds the earlier turns; re-sending them would duplicate the history."""
    runner = _Runner()
    handler = ClaudeCLIHandler(run=runner)

    _complete(handler, _messages(("user", "first")), litellm_session_id="s-resume")
    _complete(
        handler,
        _messages(("user", "first"), ("assistant", "ok"), ("user", "second")),
        litellm_session_id="s-resume",
    )

    assert runner.flag("--resume", call=1) == session_uuid("anonymous", "s-resume")
    assert runner.calls[1]["prompt"] == "second"
    assert "first" not in runner.calls[1]["prompt"]


def test_without_a_session_key_every_call_stands_alone():
    runner = _Runner()
    handler = ClaudeCLIHandler(run=runner)

    _complete(handler, _messages(("user", "hello")))

    assert "--session-id" not in runner.calls[0]["command"]
    assert "--resume" not in runner.calls[0]["command"]


def test_a_vanished_session_is_reopened_with_the_whole_conversation():
    """Claude Code's history can be cleared underneath us; the turn must still succeed."""
    runner = _Runner(
        (0, _cli_json("ok"), ""),  # turn 1 opens the session
        (1, "", "No conversation found with session ID: abc"),  # turn 2's resume finds it gone
        (0, _cli_json("recovered"), ""),  # and the retry succeeds
    )
    handler = ClaudeCLIHandler(run=runner)
    _complete(handler, _messages(("user", "first")), litellm_session_id="s-gone")

    response = _complete(
        handler,
        _messages(("user", "first"), ("assistant", "ok"), ("user", "second")),
        litellm_session_id="s-gone",
    )

    assert len(runner.calls) == 3, "the failed resume is retried once"
    assert "--session-id" in runner.calls[2]["command"]
    assert "first" in runner.calls[2]["prompt"], "the retry replays the conversation it could not resume"
    assert response.choices[0].message.content == "recovered"


# --- safety ----------------------------------------------------------------------------------------


def test_tools_are_disabled_by_default():
    """Claude Code can read files and run commands; a gateway must not hand callers that by default."""
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")))

    assert runner.flag("--disallowed-tools") == "*"
    assert "--allowed-tools" not in runner.calls[0]["command"]


def test_tools_are_enabled_only_when_a_deployment_names_them():
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")), claude_cli_allowed_tools=["Read", "Grep"])

    assert runner.flag("--allowed-tools") == "Read,Grep"
    assert "--disallowed-tools" not in runner.calls[0]["command"]


def test_the_token_never_reaches_the_child_environment():
    """A token in the environment is readable through /proc; it is passed on a file descriptor instead."""
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")), api_key="sk-ant-oat01-secret")

    env = runner.calls[0]["env"]
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "ANTHROPIC_API_KEY" not in env
    # The descriptor number is only known once the pipe exists; the runner names it (tested against a real child below).
    assert "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR" not in env
    assert "sk-ant-oat01-secret" not in json.dumps(env)
    assert runner.calls[0]["token"] == "sk-ant-oat01-secret"


def test_the_token_is_not_in_the_command_line():
    """argv is world-readable through /proc too."""
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")), api_key="sk-ant-oat01-secret")

    assert "sk-ant-oat01-secret" not in " ".join(runner.calls[0]["command"])


# --- the system prompt -----------------------------------------------------------------------------


def test_claude_codes_own_system_prompt_is_replaced():
    """Left in place it costs ~3.7k tokens and makes the model emit tool-call syntax as plain text."""
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("system", "You are terse."), ("user", "hi")))

    assert runner.flag("--system-prompt") == "You are terse."
    assert "--append-system-prompt" not in runner.calls[0]["command"]


def test_a_request_without_a_system_message_still_replaces_it():
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")))

    assert runner.flag("--system-prompt") == CLAUDE_CLI_DEFAULT_SYSTEM_PROMPT


def test_several_system_messages_are_joined():
    assert system_prompt_of(_messages(("system", "A"), ("system", "B"), ("user", "x"))) == "A\n\nB"


def test_structured_content_is_flattened():
    messages = [{"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "two"}]}]
    assert render_prompt(messages, latest_only=True) == "part one\ntwo"


def test_a_request_with_no_user_message_is_rejected():
    with pytest.raises(ClaudeCLIError):
        render_prompt(_messages(("system", "only a system prompt")), latest_only=False)


# --- reading the result ----------------------------------------------------------------------------


def test_usage_counts_cached_tokens_in_the_prompt_total():
    """A resumed conversation reads its prefix from cache; omitting it would understate the prompt."""
    usage = usage_of(parse_cli_output(_cli_json()))

    assert usage.prompt_tokens == 4 + 2000 + 100
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 2112


def test_the_response_carries_the_text_and_the_model():
    response = to_model_response(parse_cli_output(_cli_json("the answer")), "claude-sonnet-5", ModelResponse())

    assert response.choices[0].message.content == "the answer"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "claude-sonnet-5"


@pytest.mark.parametrize(
    "stop_reason,expected",
    [("end_turn", "stop"), ("max_tokens", "length"), ("tool_use", "tool_calls"), ("something_new", "stop")],
)
def test_stop_reasons_map_to_openai_vocabulary(stop_reason, expected):
    response = to_model_response(parse_cli_output(_cli_json(stop_reason=stop_reason)), "m", ModelResponse())
    assert response.choices[0].finish_reason == expected


def test_an_error_result_is_raised():
    with pytest.raises(ClaudeCLIError, match="reported an error"):
        to_model_response(parse_cli_output(_cli_json("boom", is_error=True)), "m", ModelResponse())


def test_output_that_is_not_json_is_reported_with_its_text():
    with pytest.raises(ClaudeCLIError, match="not JSON"):
        parse_cli_output("Please run /login")


def test_a_nonzero_exit_reports_stderr():
    runner = _Runner((1, "", "something went wrong"))
    with pytest.raises(ClaudeCLIError, match="something went wrong"):
        _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")))


@pytest.mark.parametrize(
    "stderr,expected",
    [
        ("No conversation found with session ID: abc", True),
        ("Session not found", True),
        ("some other failure", False),
    ],
)
def test_a_missing_session_is_recognised(stderr, expected):
    assert is_missing_session(stderr) is expected


# --- configuration ---------------------------------------------------------------------------------


def test_the_model_is_passed_without_the_provider_prefix():
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")))

    assert runner.flag("--model") == "claude-sonnet-5"


def test_a_deployment_can_point_at_its_own_config_directory():
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")), claude_cli_config_dir="~/somewhere")

    assert runner.calls[0]["env"]["CLAUDE_CONFIG_DIR"] == os.path.expanduser("~/somewhere")


def test_the_timeout_is_configurable():
    runner = _Runner()
    _complete(ClaudeCLIHandler(run=runner), _messages(("user", "hi")), claude_cli_timeout=42)

    assert runner.calls[0]["timeout"] == 42.0


# --- the async path --------------------------------------------------------------------------------


class _AsyncRunner(_Runner):
    async def __call__(self, command, env, token, timeout, prompt=""):
        return _Runner.__call__(self, command, env, token, timeout, prompt)


@pytest.mark.asyncio
async def test_the_async_path_behaves_like_the_sync_one():
    runner = _AsyncRunner()
    handler = ClaudeCLIHandler(run_async=runner)

    response = await handler.acompletion(
        model="claude_cli/claude-sonnet-5",
        messages=_messages(("user", "hi")),
        model_response=ModelResponse(),
        litellm_params={"litellm_session_id": "s-async", "api_key": "sk-ant-oat01-x"},
    )

    assert response.choices[0].message.content == "ok"
    assert runner.flag("--session-id") == session_uuid("anonymous", "s-async")
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in runner.calls[0]["env"]


# --- reporting the turn to the transcript store -----------------------------------------------------


@pytest.fixture
def reported():
    """Whatever the provider hands the transcript sync, if anything."""
    from litellm.llms.claude_cli.sync import set_transcript_sync

    collected = []
    set_transcript_sync(collected.append)
    yield collected
    set_transcript_sync(None)


def test_a_finished_turn_is_reported_for_storing(reported):
    handler = ClaudeCLIHandler(run=_Runner())

    _complete(
        handler,
        _messages(("user", "hi")),
        litellm_session_id="s-store",
        metadata={"user_api_key_hash": "hashed-key"},
        api_key="sk-ant-oat01-x",
    )

    assert len(reported) == 1
    assert reported[0].session_id == "4f0af742-dddb-4b02-95f1-ae56834c6cb2"
    assert reported[0].owner == "hashed-key"
    assert reported[0].client_session_id == "s-store"
    assert reported[0].projects_dir is None


def test_a_deployments_own_claude_home_is_where_its_transcripts_are(reported):
    handler = ClaudeCLIHandler(run=_Runner())

    _complete(handler, _messages(("user", "hi")), claude_cli_config_dir="/srv/claude", api_key="sk-ant-oat01-x")

    assert reported[0].projects_dir == "/srv/claude/projects"


def test_a_failed_turn_has_no_transcript_to_report(reported):
    handler = ClaudeCLIHandler(run=_Runner((1, "", "boom")))

    with pytest.raises(ClaudeCLIError):
        _complete(handler, _messages(("user", "hi")), api_key="sk-ant-oat01-x")

    assert reported == []


def test_storing_the_transcript_cannot_break_the_answer(reported):
    """A turn the caller paid for is not failed by a problem writing it down."""
    from litellm.llms.claude_cli.sync import set_transcript_sync

    def explode(request):
        raise RuntimeError("no database today")

    set_transcript_sync(explode)
    handler = ClaudeCLIHandler(run=_Runner())

    response = _complete(handler, _messages(("user", "hi")), api_key="sk-ant-oat01-x")

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_the_async_path_reports_its_turn_too(reported):
    handler = ClaudeCLIHandler(run_async=_AsyncRunner())

    await handler.acompletion(
        model="claude_cli/claude-sonnet-5",
        messages=_messages(("user", "hi")),
        model_response=ModelResponse(),
        litellm_params={"litellm_session_id": "s-async", "api_key": "sk-ant-oat01-x"},
    )

    assert [request.client_session_id for request in reported] == ["s-async"]


# --- streaming ------------------------------------------------------------------------------------------


from litellm.llms.claude_cli.handler import StreamExit, _run, _stream, _stream_async  # noqa: E402


def _delta(text):
    return json.dumps(
        {
            "type": "stream_event",
            "event": {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        }
    )


def _result_line(**overrides):
    return json.dumps({"type": "result", **json.loads(_cli_json(**overrides))})


class _StreamRunner:
    """Stands in for the streaming subprocess: each run yields its lines, then how it ended."""

    def __init__(self, *runs):
        self.runs = list(runs) or [([_delta("o"), _delta("k"), _result_line()], StreamExit(0, ""))]
        self.calls = []
        self.prompts = []

    def __call__(self, command, env, token, timeout, prompt=""):
        self.calls.append(list(command))
        self.prompts.append(prompt)
        lines, ended = self.runs.pop(0) if len(self.runs) > 1 else self.runs[0]
        yield from lines
        yield ended

    def flag(self, name, call=0):
        argv = self.calls[call]
        return argv[argv.index(name) + 1] if name in argv else None


class _AsyncStreamRunner(_StreamRunner):
    async def __call__(self, command, env, token, timeout, prompt=""):
        for item in _StreamRunner.__call__(self, command, env, token, timeout, prompt):
            yield item


def _streamed(handler, messages, **params):
    return list(
        handler.streaming(
            model="claude_cli/claude-sonnet-5",
            messages=messages,
            litellm_params={"api_key": "sk-ant-oat01-x", **params},
        )
    )


def test_the_reply_streams_as_it_is_written_then_says_how_it_ended():
    handler = ClaudeCLIHandler(stream=_StreamRunner())

    chunks = _streamed(handler, _messages(("user", "hi")))

    assert [chunk["text"] for chunk in chunks] == ["o", "k", ""]
    assert [chunk["is_finished"] for chunk in chunks] == [False, False, True]
    assert chunks[-1]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == {"prompt_tokens": 2104, "completion_tokens": 8, "total_tokens": 2112}


def test_streaming_asks_claude_code_for_partial_messages():
    """Without partial messages the stream is one event per whole turn, which is not streaming."""
    runner = _StreamRunner()

    _streamed(ClaudeCLIHandler(stream=runner), _messages(("user", "hi")))

    assert runner.flag("--output-format") == "stream-json"
    assert "--include-partial-messages" in runner.calls[0] and "--verbose" in runner.calls[0]


def test_bookkeeping_lines_are_not_text():
    noise = [
        json.dumps({"type": "system", "subtype": "init"}),
        json.dumps({"type": "assistant", "message": {}}),
        "not json",
        "",
    ]
    runner = _StreamRunner(([*noise, _delta("ok"), _result_line()], StreamExit(0, "")))

    chunks = _streamed(ClaudeCLIHandler(stream=runner), _messages(("user", "hi")))

    assert [chunk["text"] for chunk in chunks] == ["ok", ""]


def test_a_streamed_conversation_resumes_with_only_the_newest_message():
    runner = _StreamRunner()
    handler = ClaudeCLIHandler(stream=runner)

    _streamed(handler, _messages(("user", "one")), litellm_session_id="s-stream")
    _streamed(handler, _messages(("user", "one"), ("assistant", "ok"), ("user", "two")), litellm_session_id="s-stream")

    assert runner.flag("--resume", call=1) == session_uuid("anonymous", "s-stream")
    assert runner.prompts[1] == "two"


def test_a_vanished_session_is_started_again_with_the_whole_conversation():
    session = session_uuid("anonymous", "s-gone")
    from litellm.llms.claude_cli.handler import _STARTED_SESSIONS

    _STARTED_SESSIONS.add(session)
    runner = _StreamRunner(
        ([], StreamExit(1, "No conversation found with session ID")),
        ([_delta("fresh"), _result_line()], StreamExit(0, "")),
    )

    chunks = _streamed(
        ClaudeCLIHandler(stream=runner),
        _messages(("user", "one"), ("assistant", "a"), ("user", "two")),
        litellm_session_id="s-gone",
    )

    assert [chunk["text"] for chunk in chunks] == ["fresh", ""]
    assert runner.flag("--session-id", call=1) == session
    assert "user: one" in runner.prompts[1]


def test_a_session_started_by_an_earlier_process_is_resumed_not_restarted():
    """The started-sessions record is per process; after a restart the derived id already exists in Claude Code."""
    runner = _StreamRunner(
        ([], StreamExit(1, "Error: Session ID is already in use")),
        ([_delta("again"), _result_line()], StreamExit(0, "")),
    )

    chunks = _streamed(
        ClaudeCLIHandler(stream=runner),
        _messages(("user", "one"), ("assistant", "a"), ("user", "two")),
        litellm_session_id="s-restarted",
    )

    assert [chunk["text"] for chunk in chunks] == ["again", ""]
    assert runner.flag("--resume", call=1) == session_uuid("anonymous", "s-restarted")
    assert runner.prompts[1] == "two"


def test_a_failure_after_text_is_not_retried():
    """A second run would repeat what the caller has already been shown."""
    session = session_uuid("anonymous", "s-late")
    from litellm.llms.claude_cli.handler import _STARTED_SESSIONS

    _STARTED_SESSIONS.add(session)
    runner = _StreamRunner(([_delta("partial")], StreamExit(1, "No conversation found")))
    chunks = []

    with pytest.raises(ClaudeCLIError):
        for chunk in ClaudeCLIHandler(stream=runner).streaming(
            model="claude_cli/claude-sonnet-5",
            messages=_messages(("user", "hi")),
            litellm_params={"api_key": "sk-ant-oat01-x", "litellm_session_id": "s-late"},
        ):
            chunks.append(chunk)

    assert [chunk["text"] for chunk in chunks] == ["partial"]
    assert len(runner.calls) == 1


def test_a_failed_run_raises():
    runner = _StreamRunner(([], StreamExit(2, "boom")))

    with pytest.raises(ClaudeCLIError, match="boom"):
        _streamed(ClaudeCLIHandler(stream=runner), _messages(("user", "hi")))


def test_a_run_that_reports_an_error_raises():
    runner = _StreamRunner(([_result_line(is_error=True, result="quota exhausted")], StreamExit(0, "")))

    with pytest.raises(ClaudeCLIError, match="quota exhausted"):
        _streamed(ClaudeCLIHandler(stream=runner), _messages(("user", "hi")))


def test_a_run_that_never_reports_a_result_raises():
    runner = _StreamRunner(([_delta("hi")], StreamExit(0, "")))

    with pytest.raises(ClaudeCLIError, match="no result record"):
        _streamed(ClaudeCLIHandler(stream=runner), _messages(("user", "hi")))


def test_a_streamed_turn_is_reported_for_storing(reported):
    _streamed(ClaudeCLIHandler(stream=_StreamRunner()), _messages(("user", "hi")), litellm_session_id="s-rec")

    assert [request.client_session_id for request in reported] == ["s-rec"]


@pytest.mark.asyncio
async def test_the_async_stream_behaves_like_the_sync_one(reported):
    runner = _AsyncStreamRunner()
    handler = ClaudeCLIHandler(stream_async=runner)

    chunks = [
        chunk
        async for chunk in handler.astreaming(
            model="claude_cli/claude-sonnet-5",
            messages=_messages(("user", "hi")),
            litellm_params={"api_key": "sk-ant-oat01-x", "litellm_session_id": "s-as"},
        )
    ]

    assert [chunk["text"] for chunk in chunks] == ["o", "k", ""]
    assert runner.flag("--session-id") == session_uuid("anonymous", "s-as")
    assert [request.client_session_id for request in reported] == ["s-as"]


# --- the real subprocess runners, against a stand-in for Claude Code ---------------------------------------


@pytest.fixture
def fake_claude(tmp_path):
    """A script that behaves like `claude --print --output-format stream-json`: echoes the token it was
    handed on fd 3, writes a lot to stderr, then optionally sleeps."""

    def make(sleep=0):
        script = tmp_path / "claude"
        script.write_text(
            "#!/usr/bin/env bash\n"
            'token=$(cat <&"$CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")\n'
            "head -c 200000 /dev/zero | tr '\\0' 'e' >&2\n"
            f"echo '{_delta('token=')}'\n"
            'printf \'{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"%s"}}}\\n\' "$token"\n'
            f"sleep {sleep}\n"
            "echo done\n"
        )
        script.chmod(0o755)
        return (str(script),)

    return make


def _env():
    """What the handler hands a runner: no descriptor number yet, since only the runner knows it."""
    return {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"}


def test_the_sync_runner_streams_lines_and_does_not_stall_on_stderr(fake_claude):
    """200 KB of stderr would fill an unread pipe and hang the child."""
    items = list(_stream(fake_claude(), _env(), "secret-token", timeout=20))

    lines, ended = items[:-1], items[-1]
    assert "".join(stream_text(line) or "" for line in lines) == "token=secret-token"
    assert ended.returncode == 0 and len(ended.stderr) == 200000


@pytest.mark.asyncio
async def test_the_async_runner_streams_lines_and_does_not_stall_on_stderr(fake_claude):
    items = [item async for item in _stream_async(fake_claude(), _env(), "secret-token", timeout=20)]

    lines, ended = items[:-1], items[-1]
    assert "".join(stream_text(line) or "" for line in lines) == "token=secret-token"
    assert ended.returncode == 0 and len(ended.stderr) == 200000


def test_the_sync_runner_gives_up_after_its_timeout(fake_claude):
    with pytest.raises(ClaudeCLIError, match="timed out"):
        list(_stream(fake_claude(sleep=30), _env(), "t", timeout=1))


@pytest.mark.asyncio
async def test_the_async_runner_gives_up_after_its_timeout(fake_claude):
    with pytest.raises(ClaudeCLIError, match="timed out"):
        [item async for item in _stream_async(fake_claude(sleep=30), _env(), "t", timeout=1)]


def test_a_caller_who_stops_reading_ends_the_process(fake_claude, tmp_path):
    """A client that disconnects mid-reply must not leave Claude Code running to no one."""
    import psutil

    runner = _stream(fake_claude(sleep=30), _env(), "t", timeout=60)
    next(runner)
    children = psutil.Process().children(recursive=True)
    runner.close()

    gone, alive = psutil.wait_procs(children, timeout=5)
    assert alive == []


def test_the_one_shot_runner_delivers_the_token_where_it_says(tmp_path):
    """The non-streaming path had the same fixed-descriptor bug: Claude Code read a closed fd 3 and fell back
    to whatever login it found on its own."""
    script = tmp_path / "claude"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'token=$(cat <&"$CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")\n'
        'printf \'{"result":"%s","is_error":false}\\n\' "$token"\n'
    )
    script.chmod(0o755)
    blocker = os.open(os.devnull, os.O_RDONLY)  # hold fd 3 (or whatever is lowest) so the pipe cannot land there
    try:
        returncode, stdout, _ = _run((str(script),), _env(), "secret-token", timeout=20)
    finally:
        os.close(blocker)

    assert returncode == 0 and json.loads(stdout)["result"] == "secret-token"


def test_the_prompt_reaches_the_child_on_stdin_not_argv(tmp_path):
    """One argv string is capped at 128 KiB, and argv is readable by every local user."""
    script = tmp_path / "claude"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'cat <&"$CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR" >/dev/null\n'
        'printf \'{"result":"%s","argc":%d}\\n\' "$(wc -c)" "$#"\n'
    )
    script.chmod(0o755)
    prompt = "x" * 300_000

    returncode, stdout, _ = _run((str(script),), _env(), "t", timeout=20, prompt=prompt)

    assert returncode == 0 and json.loads(stdout) == {"result": str(len(prompt)), "argc": 0}


def test_a_deployments_claude_cli_settings_reach_the_provider():
    """LiteLLM drops any litellm_params key it has not been told about; all three were silently lost."""
    import litellm

    seen = {}

    class _Recording(ClaudeCLIHandler):
        def completion(self, *args, litellm_params=None, **kwargs):
            seen.update(litellm_params or {})
            return ModelResponse()

    litellm.custom_provider_map = [{"provider": "claude_cli", "custom_handler": _Recording()}]
    from litellm.utils import custom_llm_setup

    custom_llm_setup()
    try:
        litellm.completion(
            model="claude_cli/claude-sonnet-5",
            messages=[{"role": "user", "content": "hi"}],
            claude_cli_config_dir="/srv/claude",
            claude_cli_allowed_tools=["Read"],
            claude_cli_timeout=30,
        )
    finally:
        litellm.custom_provider_map = []

    assert seen["claude_cli_config_dir"] == "/srv/claude"
    assert seen["claude_cli_allowed_tools"] == ["Read"]
    assert seen["claude_cli_timeout"] == 30
