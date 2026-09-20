"""Mapping between LiteLLM's chat interface and Claude Code's headless JSON."""

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypedDict

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly

from litellm.types.utils import Choices, Message, ModelResponse, Usage

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


def to_model_response(result: CliResult, model: str, response: ModelResponse) -> ModelResponse:
    if result.get("is_error"):
        raise ClaudeCLIError(f"Claude CLI reported an error: {result.get('result') or result.get('subtype')}")
    text: Final = result.get("result") or ""
    stop_reason: Final = result.get("stop_reason")
    response.choices = [  # mutable-ok: the single choice this interface returns
        Choices(
            finish_reason=_FINISH_REASONS.get(stop_reason or "", "stop"),
            index=0,
            message=Message(content=text, role="assistant"),
        )
    ]
    response.model = model
    setattr(response, "usage", usage_of(result))  # noqa: B010  # ModelResponse declares usage dynamically
    return response
