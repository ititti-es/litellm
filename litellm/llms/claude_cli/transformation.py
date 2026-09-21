"""Mapping between LiteLLM's chat interface and Claude Code's headless JSON."""

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly

from litellm.types.llms.openai import ChatCompletionUsageBlock
from litellm.types.utils import Choices, GenericStreamingChunk, Message, ModelResponse, Usage

from .common_utils import CLAUDE_CLI_DEFAULT_SYSTEM_PROMPT, ClaudeCLIError


class _CliUsage(TypedDict):
    input_tokens: NotRequired[ReadOnly[int]]
    output_tokens: NotRequired[ReadOnly[int]]
    cache_read_input_tokens: NotRequired[ReadOnly[int]]
    cache_creation_input_tokens: NotRequired[ReadOnly[int]]


class CliResult(TypedDict):
    result: NotRequired[ReadOnly[str]]
    session_id: NotRequired[ReadOnly[str]]
    is_error: NotRequired[ReadOnly[bool]]
    subtype: NotRequired[ReadOnly[str]]
    stop_reason: NotRequired[ReadOnly[str | None]]
    total_cost_usd: NotRequired[ReadOnly[float]]
    usage: NotRequired[ReadOnly[_CliUsage]]


_RESULT_ADAPTER: Final = TypeAdapter(CliResult)
_NO_USAGE: Final[_CliUsage] = {}  # mutable-ok: stands in for a result that reported no usage

# Claude Code reports why it stopped in its own vocabulary; these are the OpenAI equivalents.
_FINISH_REASONS: Final = MappingProxyType(
    {  # mutable-ok: module constant, read-only at runtime
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
)


def _text_of(content: object) -> str:
    """Flatten a message's content, which may be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []  # mutable-ok: accumulator, returned joined
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def system_prompt_of(messages: Sequence[Mapping[str, object]]) -> str:
    """Return the system prompt to hand Claude Code.

    Replacing its own prompt rather than appending to it is deliberate: Claude Code's describes an agent
    with tools, which both costs tokens on every conversation and makes the model emit tool-call syntax as
    text once tools are disabled.
    """
    collected: Final = tuple(_text_of(m.get("content")) for m in messages if m.get("role") == "system")
    joined: Final = "\n\n".join(text for text in collected if text)
    return joined or CLAUDE_CLI_DEFAULT_SYSTEM_PROMPT


def render_prompt(messages: Sequence[Mapping[str, object]], latest_only: bool) -> str:
    """Render the conversation, or just its last exchange when Claude Code already holds the rest.

    On a resumed session, everything before the final user message is already in Claude Code's own
    transcript; re-sending it would duplicate the history rather than continue it.
    """
    conversation: Final = tuple(m for m in messages if m.get("role") != "system")
    if not conversation:
        raise ClaudeCLIError("No user message to send", status_code=400)
    if latest_only:
        return _text_of(conversation[-1].get("content"))
    return "\n\n".join(
        f"{m.get('role', 'user')}: {_text_of(m.get('content'))}" for m in conversation if _text_of(m.get("content"))
    )


def parse_cli_output(stdout: str) -> CliResult:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ClaudeCLIError(f"Claude CLI returned output that was not JSON: {stdout[:400]}") from exc
    if not isinstance(payload, dict):
        raise ClaudeCLIError("Claude CLI returned JSON that was not an object")
    try:
        return _RESULT_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ClaudeCLIError(f"Claude CLI returned unexpected fields: {exc}") from exc


def usage_of(result: CliResult) -> Usage:
    """Map Claude Code's usage to LiteLLM's.

    Cache reads are counted in the prompt total, matching how the Anthropic provider reports them, so a
    resumed conversation does not appear to have arrived from nowhere.
    """
    reported: Final = result.get("usage") or _NO_USAGE
    cached: Final = int(reported.get("cache_read_input_tokens") or 0)
    created: Final = int(reported.get("cache_creation_input_tokens") or 0)
    prompt_tokens: Final = int(reported.get("input_tokens") or 0) + cached + created
    completion_tokens: Final = int(reported.get("output_tokens") or 0)
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def finish_reason_of(result: CliResult) -> str:
    return _FINISH_REASONS.get(result.get("stop_reason") or "", "stop")


def raise_if_error(result: CliResult) -> None:
    if result.get("is_error"):
        raise ClaudeCLIError(f"Claude CLI reported an error: {result.get('result') or result.get('subtype')}")


def to_model_response(result: CliResult, model: str, response: ModelResponse) -> ModelResponse:
    raise_if_error(result)
    text: Final = result.get("result") or ""
    response.choices = [  # mutable-ok: the single choice this interface returns
        Choices(
            finish_reason=finish_reason_of(result),
            index=0,
            message=Message(content=text, role="assistant"),
        )
    ]
    response.model = model
    setattr(response, "usage", usage_of(result))  # noqa: B010  # ModelResponse declares usage dynamically
    return response


# --- streaming (`--output-format stream-json --include-partial-messages`) --------------------------------------
#
# One JSON object per line. Text arrives as `stream_event` records wrapping the Messages API's own
# `content_block_delta` / `text_delta` events, and the run ends with a `result` record of exactly the shape the
# non-streaming `json` format returns. Everything else (system, assistant, rate-limit records) is bookkeeping.

_STREAM_EVENT: Final = "stream_event"
_RESULT: Final = "result"


def _line_object(line: str) -> Mapping[str, object] | None:
    text: Final = line.strip()
    if not text.startswith("{"):
        return None
    try:
        payload: Final = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def stream_text(line: str) -> str | None:
    """The text a stream line adds to the reply, if it adds any."""
    record: Final = _line_object(line)
    if record is None or record.get("type") != _STREAM_EVENT:
        return None
    event: Final = record.get("event")
    if not isinstance(event, Mapping) or event.get("type") != "content_block_delta":
        return None
    delta: Final = event.get("delta")
    if not isinstance(delta, Mapping) or delta.get("type") != "text_delta":
        return None
    text: Final = delta.get("text")
    return text if isinstance(text, str) and text else None


def stream_result(line: str) -> CliResult | None:
    """The run's closing record, if this line is it."""
    record: Final = _line_object(line)
    if record is None or record.get("type") != _RESULT:
        return None
    try:
        return _RESULT_ADAPTER.validate_python(record)
    except ValidationError as exc:
        raise ClaudeCLIError(f"Claude CLI returned an unexpected result record: {exc}") from exc


def text_chunk(text: str) -> GenericStreamingChunk:
    return {  # mutable-ok: GenericStreamingChunk is the TypedDict CustomLLM streaming requires
        "text": text,
        "is_finished": False,
        "finish_reason": "",
        "usage": None,
        "index": 0,
        "tool_use": None,
    }


def final_chunk(result: CliResult) -> GenericStreamingChunk:
    """The closing chunk: why the reply stopped and what it cost. Carries no text of its own."""
    raise_if_error(result)
    usage: Final = usage_of(result)
    block: Final[ChatCompletionUsageBlock] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    return {  # mutable-ok: GenericStreamingChunk is the TypedDict CustomLLM streaming requires
        "text": "",
        "is_finished": True,
        "finish_reason": finish_reason_of(result),
        "usage": block,
        "index": 0,
        "tool_use": None,
    }
