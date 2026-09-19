"""What a compaction asks for, and how its summary is put back into the session

A policy is plain data so it can live in the database (`LiteLLM_CompactionPolicy`) and be edited without code:

- `instructions`: the user message appended after the cached history, asking for a summary
- `settings` (all optional):
  - `update_instructions`: used instead when the history already starts from an earlier summary
  - `summary_role`: "user" (default) or "assistant", the role of the message that carries the summary
  - `summary_lead_in`: a user message placed before an assistant summary (opencode asks "What did we do so far?")
  - `summary_prefix`, `summary_suffix`: text wrapped around the summary
  - `keep_turns`: how many of the latest user turns are kept verbatim after the summary (default 1)
  - `max_output_tokens`: a cap for the summary call

`{{date}}` in any text is replaced with the current UTC date. The summary call always reuses the turn's cached
prefix (model, instructions, tools), so a policy changes only what is asked and how the answer is framed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from litellm.sessions.session import Item

DATE_PLACEHOLDER: Final = "{{date}}"


class PolicySettings(BaseModel):
    """The `settings` column of a stored policy"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    update_instructions: str | None = None
    summary_role: Literal["user", "assistant"] = "user"
    summary_lead_in: str | None = None
    summary_prefix: str = ""
    summary_suffix: str = ""
    keep_turns: Annotated[int, Field(ge=1)] = 1
    max_output_tokens: Annotated[int | None, Field(ge=256)] = None


class PolicyDocument(BaseModel):
    """The editable JSON form of a policy"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, Field(min_length=1)]
    source: str = "custom"
    instructions: Annotated[str, Field(min_length=1)]
    settings: PolicySettings = PolicySettings()


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    name: str
    instructions: str
    settings: PolicySettings = field(default_factory=PolicySettings)
    source: str = "custom"

    def request_text(self, updating: bool, now: datetime | None = None) -> str:
        text: Final = self.settings.update_instructions if updating and self.settings.update_instructions else None
        return render(text or self.instructions, now)

    def summary_items(self, summary: str, now: datetime | None = None) -> tuple[tuple[Item, ...], int]:
        """The items that carry *summary* into the rebased session, and the position of the one holding it"""
        body: Final = render(self.settings.summary_prefix, now) + summary + render(self.settings.summary_suffix, now)
        if self.settings.summary_role == "user":
            return (_message("user", "input_text", body),), 0
        lead_in: Final = self.settings.summary_lead_in
        if lead_in:
            return (_message("user", "input_text", render(lead_in, now)), _message("assistant", "output_text", body)), 1
        return (_message("assistant", "output_text", body),), 0

    def to_json(self) -> str:
        """The editable form: `name`, `source`, `instructions` and every setting, defaults included"""
        document: Final = PolicyDocument(
            name=self.name, source=self.source, instructions=self.instructions, settings=self.settings
        )
        return document.model_dump_json(indent=2)

    def document(self) -> Mapping[str, object]:
        """The same form as a plain mapping, for any serialisation"""
        return PolicyDocument(
            name=self.name, source=self.source, instructions=self.instructions, settings=self.settings
        ).model_dump()


def policy_from(name: str, source: str, instructions: str, settings: object) -> CompactionPolicy:
    """A policy from its stored columns; `settings` may be JSON text, a mapping, or absent"""
    raw: Final = json.loads(settings) if isinstance(settings, str) else settings
    return CompactionPolicy(
        name=name,
        instructions=instructions,
        settings=PolicySettings.model_validate(raw) if isinstance(raw, Mapping) else PolicySettings(),
        source=source,
    )


def policy_from_document(raw: object) -> CompactionPolicy:
    """The inverse of `document` (and, after parsing, of `to_json`)"""
    document: Final = PolicyDocument.model_validate(raw)
    return CompactionPolicy(document.name, document.instructions, document.settings, document.source)


def policy_from_json(text: str) -> CompactionPolicy:
    return policy_from_document(json.loads(text))


def render(text: str, now: datetime | None = None) -> str:
    return text.replace(DATE_PLACEHOLDER, (now or datetime.now(UTC)).strftime("%Y-%m-%d"))


def _message(role: str, part: str, text: str) -> Item:
    return {"type": "message", "role": role, "content": [{"type": part, "text": text}]}  # mutable-ok: JSON input item
