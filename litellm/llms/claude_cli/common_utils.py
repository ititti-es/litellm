"""Constants and helpers for running Claude Code as an inference backend.

Anthropic issues Claude Pro/Max OAuth tokens to Claude Code and refuses them on `/v1/messages` directly:
a request there comes back as a generic `rate_limit_error` even with quota to spare. The supported way to
spend a subscription is to let Claude Code make the call, which is what this provider does -- the same
approach openclaw takes, and using Claude Code's documented headless mode rather than impersonating it.
"""

import os
import uuid
from typing import Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException

CLAUDE_CLI_PROVIDER: Final = "claude_cli"
CLAUDE_CLI_DEFAULT_BINARY: Final = "claude"
CLAUDE_CLI_DEFAULT_TIMEOUT: Final = 600.0

# Claude Code's own system prompt describes its tools and agent role. Left in place it costs ~3.7k tokens
# per conversation and, with tools disabled, makes the model emit tool-call syntax as plain text. Replacing
# it is what turns Claude Code into a general-purpose model endpoint.
CLAUDE_CLI_DEFAULT_SYSTEM_PROMPT: Final = (
    "You are a helpful assistant. You have no tools available. Answer directly in plain text."
)

# Tool access is off unless a deployment asks for it. Claude Code can read files and run shell commands, so
# a gateway that left them on would hand every caller code execution on the proxy host.
CLAUDE_CLI_NO_TOOLS: Final = "*"

# The name of the environment variable Claude Code reads a file descriptor number from. Passing the token
# this way keeps it out of the child's environment, where /proc would expose it. The number is set per call
# to wherever the pipe actually is; see _TokenPipe.child_env.
CLAUDE_CLI_TOKEN_FD_ENV: Final = "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"

# Session ids are UUIDs, and Claude Code accepts one we choose. Deriving it from the caller's session means
# no mapping table has to be kept anywhere.
CLAUDE_CLI_SESSION_NAMESPACE: Final = uuid.UUID("6f1d4f9c-2a9e-4c43-9a7f-1b2c3d4e5f60")

_MISSING_SESSION_MARKERS: Final = ("no conversation found", "session not found")


class ClaudeCLIError(BaseLLMException):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(status_code=status_code, message=message)


def binary_path() -> str:
    return os.getenv("CLAUDE_CLI_PATH") or CLAUDE_CLI_DEFAULT_BINARY


def session_uuid(owner: str, session_key: str) -> str:
    """Return the Claude Code session id for a caller's conversation.

    Derived rather than stored, so a restart resumes the same conversation, and namespaced by owner so two
    API keys can never land on one another's session by choosing the same id.
    """
    return str(uuid.uuid5(CLAUDE_CLI_SESSION_NAMESPACE, f"{owner}\0{session_key}"))


def is_missing_session(stderr: str) -> bool:
    """Whether a failure means the session to resume does not exist yet."""
    lowered: Final = stderr.lower()
    return any(marker in lowered for marker in _MISSING_SESSION_MARKERS)
