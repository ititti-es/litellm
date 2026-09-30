"""Persistent Claude Code process with a private external MCP tool relay."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import signal
import sys
import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypedDict

import jsonschema
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm.types.utils import Choices, GenericStreamingChunk, Message, ModelResponse, Usage

from .common_utils import ClaudeCLIError, binary_path, session_uuid
from .transformation import final_chunk, render_prompt, stream_result, system_prompt_of, text_chunk

_MAX_SESSIONS: Final = 32
_MAX_RELAY_LINE: Final = 1024 * 1024
_MCP_SERVER_NAME: Final = "litellm_bridge"
_SESSIONS: Final[dict[str, BridgeSession]] = {}  # mutable-ok: process-local live subprocess registry
_START_LOCK: Final = asyncio.Lock()


class ToolDefinition(TypedDict):
    name: ReadOnly[str]
    description: ReadOnly[str]
    parameters: ReadOnly[Mapping[str, object]]


class _ObjectSchema(TypedDict):
    type: ReadOnly[str]


class _McpServer(TypedDict):
    command: ReadOnly[str]
    args: ReadOnly[tuple[str, ...]]


class _McpConfig(TypedDict):
    mcpServers: ReadOnly[Mapping[str, _McpServer]]


class _RelayResponse(TypedDict):
    content: ReadOnly[str]


class _UserMessage(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]


class _UserEvent(TypedDict):
    type: ReadOnly[str]
    message: ReadOnly[_UserMessage]


class _CompletionToolCall(TypedDict):
    id: ReadOnly[str]
    type: ReadOnly[str]
    function: ReadOnly[object]


class _FunctionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str
    description: str = ""
    parameters: Mapping[str, object] | None = None
    strict: bool | None = None


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["function"]
    function: _FunctionInput


class _ToolBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    type: Literal["tool_use"]
    id: str
    name: str
    input: Mapping[str, object]


class _RelayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capability: str
    id: str
    name: str
    arguments: Mapping[str, object]


_TOOLS_ADAPTER: Final = TypeAdapter(tuple[_ToolInput, ...])
_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, object])
_SEQUENCE_ADAPTER: Final = TypeAdapter(tuple[object, ...])


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EventOutcome:
    chunks: tuple[GenericStreamingChunk, ...]
    terminal: bool


def definitions_of(raw: object) -> tuple[ToolDefinition, ...]:
    try:
        supplied: Final = _TOOLS_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ClaudeCLIError("only function tools are supported", status_code=400) from exc
    if not supplied:
        raise ClaudeCLIError("function tools must be a nonempty array", status_code=400)
    definitions: Final = tuple(_definition(item) for item in supplied)
    if len(frozenset(item["name"] for item in definitions)) != len(definitions):
        raise ClaudeCLIError("function tool names must be unique", status_code=400)
    return definitions


def _definition(raw: _ToolInput) -> ToolDefinition:
    function: Final = raw.function
    name: Final = function.name
    default_schema: Final[_ObjectSchema] = {"type": "object"}
    schema: Final = function.parameters or default_schema
    if not name or len(name) > 128 or not all(c.isalnum() or c == "_" for c in name):
        raise ClaudeCLIError("invalid function tool name", status_code=400)
    if schema.get("type") != "object":
        raise ClaudeCLIError("function parameters must be an object schema", status_code=400)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        raise ClaudeCLIError("invalid function tool schema", status_code=400) from exc
    definition: Final[ToolDefinition] = {
        "name": name,
        "description": function.description,
        "parameters": schema,
    }
    return definition


def _tool_content(content: object) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, separators=(",", ":")) if content is not None else ""


def _results(messages: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (call_id, _tool_content(message.get("content")))
        for message in messages
        if message.get("role") == "tool" and isinstance(call_id := message.get("tool_call_id"), str)
    )


def _session_key(params: Mapping[str, object]) -> str | None:
    metadata: Final = _mapping(params.get("metadata"))
    candidates: Final = (params, metadata) if metadata is not None else (params,)
    return next(
        (
            value
            for source in candidates
            for name in ("litellm_session_id", "session_id")
            if isinstance(value := source.get(name), str) and value
        ),
        None,
    )


def _owner(params: Mapping[str, object], token: str) -> str:
    metadata: Final = _mapping(params.get("metadata"))
    if metadata is not None:
        owner: Final = next(
            (
                value
                for name in ("user_api_key_hash", "user_api_key")
                if isinstance(value := metadata.get(name), str) and value
            ),
            None,
        )
        if owner is not None:
            return owner
    return hashlib.sha256(token.encode()).hexdigest()


def _assistant_calls(event: Mapping[str, object]) -> tuple[ToolCall, ...]:
    message: Final = _mapping(event.get("message"))
    if message is None:
        return ()
    try:
        content: Final = _SEQUENCE_ADAPTER.validate_python(message.get("content"))
        blocks: Final = tuple(_ToolBlock.model_validate(block) for block in content if _is_tool_block(block))
    except ValidationError as exc:
        raise ClaudeCLIError("Claude returned malformed tool calls", status_code=502) from exc
    calls: Final = tuple(
        ToolCall(id=block.id, name=_external_tool_name(block.name), arguments=block.input) for block in blocks
    )
    if len(frozenset(call.id for call in calls)) != len(calls):
        raise ClaudeCLIError("Claude returned malformed tool calls", status_code=502)
    return calls


def _external_tool_name(name: str) -> str:
    prefix: Final = f"mcp__{_MCP_SERVER_NAME}__"
    return name.removeprefix(prefix)


def _mapping(raw: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(raw)
    except ValidationError:
        return None


def _is_tool_block(raw: object) -> bool:
    block: Final = _mapping(raw)
    return block is not None and block.get("type") == "tool_use"


class BridgeSession:
    def __init__(self, key: str, fingerprint: str, definitions: tuple[ToolDefinition, ...], timeout: float) -> None:
        self.key: Final = key
        self.fingerprint: Final = fingerprint
        self.definitions: Final = definitions
        self.timeout: Final = timeout
        self.directory: tempfile.TemporaryDirectory[str] | None = None
        self.server: asyncio.AbstractServer | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.events: Final[asyncio.Queue[Mapping[str, object]]] = asyncio.Queue()
        self.calls: Mapping[str, ToolCall] = MappingProxyType({})
        self.announced: Mapping[str, ToolCall] = MappingProxyType({})
        self.results: Mapping[str, str] = MappingProxyType({})
        self.call_ready: Final = asyncio.Condition()
        self.waiting: Mapping[str, asyncio.Future[str]] = MappingProxyType({})
        self.pending: frozenset[str] = frozenset()
        self.seen: frozenset[str] = frozenset()
        self.lock: Final = asyncio.Lock()
        self.tasks: tuple[asyncio.Task[None], ...] = ()
        self.expiry: asyncio.TimerHandle | None = None
        self.closed: bool = False
        self.capability: Final = secrets.token_urlsafe(32)

    async def start(self, model: str, system: str, prompt: str, env: Mapping[str, str], token: str) -> None:
        from .handler import (
            _STREAM_LINE_LIMIT,  # pyright: ignore[reportPrivateUsage]  # shared provider subprocess limit
            _TokenPipe,  # pyright: ignore[reportPrivateUsage]  # shared provider credential transport
        )

        directory: Final = tempfile.TemporaryDirectory(prefix="claude-tool-bridge-")
        os.chmod(directory.name, 0o700)
        self.directory = directory
        base: Final = Path(directory.name)
        (base / "system.txt").write_text(system, encoding="utf-8")
        (base / "tools.json").write_text(json.dumps(self.definitions), encoding="utf-8")
        (base / "capability").write_text(self.capability, encoding="utf-8")
        config: Final[_McpConfig] = {
            "mcpServers": {
                _MCP_SERVER_NAME: {
                    "command": sys.executable,
                    "args": (
                        "-m",
                        "litellm.llms.claude_cli.mcp_sidecar",
                        "--socket",
                        str(base / "relay.sock"),
                        "--directory",
                        directory.name,
                    ),
                }
            }
        }
        (base / "mcp.json").write_text(json.dumps(config), encoding="utf-8")
        self.server = await asyncio.start_unix_server(self._relay, path=str(base / "relay.sock"), limit=_MAX_RELAY_LINE)
        os.chmod(base / "relay.sock", 0o600)
        allowed: Final = ",".join(f"mcp__{_MCP_SERVER_NAME}__{definition['name']}" for definition in self.definitions)
        command: Final = (
            binary_path(),
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--input-format",
            "stream-json",
            "--system-prompt-file",
            str(base / "system.txt"),
            "--model",
            model,
            "--tools",
            "",
            "--restricted",
            "--strict-mcp-config",
            "--mcp-config",
            str(base / "mcp.json"),
            "--allowed-tools",
            allowed,
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
        )
        with _TokenPipe(token) as pipe:
            try:
                self.process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=pipe.child_env(env),
                    pass_fds=(pipe.read_fd,),
                    start_new_session=True,
                    limit=_STREAM_LINE_LIMIT,
                )
            except FileNotFoundError as exc:
                await self.close()
                raise ClaudeCLIError(
                    f"Claude CLI not found at {command[0]!r}. Install Claude Code or set CLAUDE_CLI_PATH.",
                    status_code=500,
                ) from exc
            except BaseException:
                await self.close()
                raise
            pipe.close_parent_read()
            pipe.deliver()
        self.tasks = (asyncio.create_task(self._read()), asyncio.create_task(self._stderr()))
        await self.send(prompt)
        self.refresh_expiry()

    async def _read(self) -> None:
        process: Final = self.process
        if process is None or process.stdout is None:
            return
        try:
            while line := await process.stdout.readline():
                try:
                    event = _MAPPING_ADAPTER.validate_json(  # rebind-ok: each iteration reads a new event
                        line
                    )
                except ValidationError:
                    continue
                await self.events.put(event)
        finally:
            await self.events.put(MappingProxyType({"type": "bridge_exit"}))
            asyncio.create_task(self.close())

    async def _stderr(self) -> None:
        process: Final = self.process
        if process is not None and process.stderr is not None:
            while await process.stderr.read(65536):
                pass

    async def _relay(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line: Final = await asyncio.wait_for(reader.readline(), 15)
            if not line or len(line) > _MAX_RELAY_LINE:
                return
            request: Final = _RelayRequest.model_validate_json(line)
            if request.capability != self.capability:
                return
            name: Final = request.name
            call_id: Final = request.id
            arguments: Final = request.arguments
            definition: Final = next((item for item in self.definitions if item["name"] == name), None)
            if definition is None or not call_id.startswith("toolu_") or call_id in self.seen:
                return
            jsonschema.validate(arguments, definition["parameters"])
            waiter: Final[asyncio.Future[str]] = asyncio.get_running_loop().create_future()
            async with self.call_ready:
                announced: Final = self.announced.get(call_id)
                if self.announced and announced is None:
                    raise ValueError("Claude announced no matching tool call")
                if announced is not None and (announced.name != name or announced.arguments != arguments):
                    raise ValueError("Claude and MCP tool calls differ")
                self.seen = self.seen | frozenset((call_id,))
                self.waiting = MappingProxyType({**self.waiting, call_id: waiter})
                self.calls = MappingProxyType({**self.calls, call_id: ToolCall(call_id, name, arguments)})
                self.call_ready.notify_all()
                stored: Final = self.results.get(call_id)
                if stored is not None:
                    self.results = MappingProxyType(
                        {key: content for key, content in self.results.items() if key != call_id}
                    )
                    self.waiting = MappingProxyType(
                        {key: pending for key, pending in self.waiting.items() if key != call_id}
                    )
                    waiter.set_result(stored)
            content: Final = await waiter
            response: Final[_RelayResponse] = {"content": content}
            writer.write((json.dumps(response) + "\n").encode())
            await writer.drain()
            async with self.call_ready:
                self.calls = MappingProxyType({key: call for key, call in self.calls.items() if key != call_id})
                self.announced = MappingProxyType({key: call for key, call in self.announced.items() if key != call_id})
        except (ValueError, jsonschema.ValidationError, OSError, TimeoutError, asyncio.CancelledError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _announce_calls(self, announced: tuple[ToolCall, ...]) -> tuple[ToolCall, ...]:
        async with self.call_ready:
            if self.pending or self.announced:
                raise ClaudeCLIError("tool bridge process state was lost", status_code=409)
            self.announced = MappingProxyType({call.id: call for call in announced})
            if not frozenset(self.calls).issubset(self.announced):
                raise ClaudeCLIError("Claude and MCP tool calls differ", status_code=502)
            relayed: Final = tuple(self.calls[call.id] for call in announced if call.id in self.calls)
        if any(
            expected.name != actual.name or expected.arguments != actual.arguments
            for actual in relayed
            if (expected := self.announced[actual.id])
        ):
            raise ClaudeCLIError("Claude and MCP tool calls differ", status_code=502)
        return announced

    async def send(self, text: str) -> None:
        process: Final = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise ClaudeCLIError("tool bridge process state was lost", status_code=409)
        try:
            event: Final[_UserEvent] = {"type": "user", "message": {"role": "user", "content": text}}
            process.stdin.write((json.dumps(event) + "\n").encode())
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise ClaudeCLIError("tool bridge process state was lost", status_code=409) from exc

    def refresh_expiry(self) -> None:
        if self.expiry is not None:
            self.expiry.cancel()
        self.expiry = asyncio.get_running_loop().call_later(self.timeout, lambda: asyncio.create_task(self.close()))

    async def _accept_results(self, results: tuple[tuple[str, str], ...]) -> bool:
        if not self.pending:
            return False
        provided: Final = tuple(item for item in results if item[0] in self.pending)
        if len(provided) != len(self.pending) or frozenset(item[0] for item in provided) != self.pending:
            raise ClaudeCLIError("tool result IDs do not match pending Claude calls", status_code=409)
        provided_ids: Final = frozenset(call_id for call_id, _ in provided)
        async with self.call_ready:
            waiters: Final = tuple((self.waiting.get(call_id), content) for call_id, content in provided)
            self.results = MappingProxyType({**self.results, **MappingProxyType(dict(provided))})
            self.waiting = MappingProxyType(
                {call_id: waiter for call_id, waiter in self.waiting.items() if call_id not in provided_ids}
            )
            self.pending = frozenset()
            self.call_ready.notify_all()
        for waiter, content in waiters:
            if waiter is not None and not waiter.done():
                waiter.set_result(content)
        return True

    async def _event_outcome(self, event: Mapping[str, object]) -> _EventOutcome:
        kind: Final = event.get("type")
        if kind == "bridge_exit":
            raise ClaudeCLIError("tool bridge process exited before completing turn", status_code=502)
        if kind == "stream_event":
            wrapped: Final = _mapping(event.get("event"))
            delta: Final = _mapping(wrapped.get("delta")) if wrapped is not None else None
            text: Final = delta.get("text") if delta is not None and delta.get("type") == "text_delta" else None
            return (
                _EventOutcome((text_chunk(text),), False)
                if isinstance(text, str) and text
                else _EventOutcome((), False)
            )
        if kind == "assistant":
            announced: Final = _assistant_calls(event)
            if not announced:
                return _EventOutcome((), False)
            relayed: Final = await self._announce_calls(announced)
            self.pending = frozenset(call.id for call in relayed)
            self.refresh_expiry()
            tool_chunks: Final = tuple(self._tool_chunk(call, index) for index, call in enumerate(relayed))
            boundary: Final[GenericStreamingChunk] = {
                "text": "",
                "is_finished": True,
                "finish_reason": "tool_calls",
                "usage": None,
                "index": 0,
                "tool_use": None,
            }
            return _EventOutcome((*tool_chunks, boundary), True)
        if kind == "result":
            result: Final = stream_result(json.dumps(event))
            if result is None:
                raise ClaudeCLIError("invalid Claude result boundary", status_code=502)
            async with self.call_ready:
                waiters: Final = tuple(self.waiting.values())
                self.calls = MappingProxyType({})
                self.announced = MappingProxyType({})
                self.results = MappingProxyType({})
                self.waiting = MappingProxyType({})
                self.pending = frozenset()
                self.seen = frozenset()
            for waiter in waiters:
                if not waiter.done():
                    waiter.cancel()
            self.refresh_expiry()
            return _EventOutcome((final_chunk(result),), True)
        return _EventOutcome((), False)

    @staticmethod
    def _tool_chunk(call: ToolCall, index: int) -> GenericStreamingChunk:
        chunk: Final[GenericStreamingChunk] = {
            "text": "",
            "is_finished": False,
            "finish_reason": "",
            "usage": None,
            "index": 0,
            "tool_use": {
                "id": call.id,
                "index": index,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, separators=(",", ":")),
                },
            },
        }
        return chunk

    async def run(
        self, results: tuple[tuple[str, str], ...], prompt: str | None
    ) -> AsyncIterator[GenericStreamingChunk]:
        if self.closed or self.lock.locked():
            raise ClaudeCLIError("tool bridge process is lost or busy", status_code=409)
        async with self.lock:
            resumed: Final = await self._accept_results(results)
            if self.expiry is not None:
                self.expiry.cancel()
                self.expiry = None
            if resumed and prompt is not None:
                raise ClaudeCLIError("tool results cannot be sent to Claude stdin", status_code=500)
            if not resumed and prompt is not None:
                await self.send(prompt)
            try:
                async with asyncio.timeout(self.timeout):
                    while True:
                        outcome = await self._event_outcome(  # rebind-ok: each loop iteration consumes a new event
                            await self.events.get()
                        )
                        for chunk in outcome.chunks:
                            yield chunk
                        if outcome.terminal:
                            return
            except TimeoutError as exc:
                raise ClaudeCLIError("tool bridge turn timed out", status_code=504) from exc

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.expiry is not None:
            self.expiry.cancel()
        for waiter in self.waiting.values():
            if not waiter.done():
                waiter.cancel()
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.process.wait()
        current: Final = asyncio.current_task()
        pending_tasks: Final = tuple(task for task in self.tasks if task is not current and not task.done())
        for task in pending_tasks:
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        if self.directory is not None:
            self.directory.cleanup()
        if _SESSIONS.get(self.key) is self:
            _SESSIONS.pop(self.key, None)


async def stream_bridge(
    model: str,
    messages: Sequence[Mapping[str, object]],
    params: Mapping[str, object],
    token: str,
    timeout: float,
    tools: object,
) -> AsyncIterator[GenericStreamingChunk]:
    definitions: Final = definitions_of(tools)
    client_session: Final = _session_key(params)
    if client_session is None:
        raise ClaudeCLIError("function tools require litellm_session_id", status_code=400)
    owner: Final = _owner(params, token)
    key: Final = session_uuid(owner, client_session)
    provider_model: Final = model.split("/", 1)[-1]
    system: Final = system_prompt_of(messages)
    fingerprint: Final = hashlib.sha256(
        json.dumps((owner, key, provider_model, system, definitions), sort_keys=True).encode()
    ).hexdigest()
    supplied: Final = _results(messages)
    async with _START_LOCK:
        existing: Final = _SESSIONS.get(key)
        if existing is None:
            if supplied:
                raise ClaudeCLIError("tool bridge process state was lost", status_code=409)
            if len(_SESSIONS) >= _MAX_SESSIONS:
                raise ClaudeCLIError("tool bridge capacity reached", status_code=503)
            session = BridgeSession(  # rebind-ok: mutually exclusive new/existing session selection
                key, fingerprint, definitions, timeout
            )
            _SESSIONS[key] = session
            try:
                from .handler import _child_env  # pyright: ignore[reportPrivateUsage]  # provider child isolation

                await session.start(provider_model, system, render_prompt(messages, False), _child_env(params), token)
            except BaseException:
                await session.close()
                raise
        else:
            if existing.closed:
                raise ClaudeCLIError("tool bridge process state was lost", status_code=409)
            if existing.fingerprint != fingerprint:
                raise ClaudeCLIError("tool bridge session configuration mismatch", status_code=409)
            session = existing  # rebind-ok: mutually exclusive new/existing session selection
    prompt: Final = render_prompt(messages, True) if existing is not None and not session.pending else None
    try:
        async for chunk in session.run(supplied, prompt):
            yield chunk
    except ClaudeCLIError as exc:
        if exc.status_code != 409:
            await session.close()
        raise
    except BaseException:
        await session.close()
        raise


async def complete_bridge(
    model: str,
    messages: Sequence[Mapping[str, object]],
    params: Mapping[str, object],
    token: str,
    timeout: float,
    tools: object,
    response: ModelResponse,
) -> ModelResponse:
    chunks: Final = [  # mutable-ok: async comprehensions only materialize as mutable containers
        chunk async for chunk in stream_bridge(model, messages, params, token, timeout, tools)
    ]
    text: Final = tuple(chunk["text"] for chunk in chunks if chunk.get("text"))
    calls: Final = [  # mutable-ok: Message requires a list at its API boundary
        _completion_tool_call(tool) for chunk in chunks if (tool := chunk.get("tool_use")) is not None
    ]
    finished: Final = tuple(chunk for chunk in chunks if chunk.get("is_finished"))
    if not finished:
        raise ClaudeCLIError("Claude stream ended without a result", status_code=502)
    choices: Final = [  # mutable-ok: ModelResponse requires a mutable list at its API boundary
        Choices(
            index=0,
            finish_reason="tool_calls" if calls else finished[-1]["finish_reason"],
            message=Message(
                role="assistant",
                content=("".join(text) or None) if calls else "".join(text),
                tool_calls=calls or None,
            ),
        )
    ]
    response.choices = choices  # rebind-ok: CustomLLM supplies the response object to populate
    reported: Final = finished[-1].get("usage")
    if reported is not None:
        setattr(  # noqa: B010  # ModelResponse declares usage dynamically
            response, "usage", Usage(**reported)
        )  # rebind-ok: CustomLLM supplies the response object to populate
    response.model = model  # rebind-ok: CustomLLM supplies the response object to populate
    return response


def _completion_tool_call(tool: object) -> _CompletionToolCall:
    raw: Final = _mapping(tool)
    call_id: Final = raw.get("id") if raw is not None else None
    function: Final = raw.get("function") if raw is not None else None
    if not isinstance(call_id, str) or function is None:
        raise ClaudeCLIError("Claude stream contained a malformed tool call", status_code=502)
    return _CompletionToolCall(id=call_id, type="function", function=function)
