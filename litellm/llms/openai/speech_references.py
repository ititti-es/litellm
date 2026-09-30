"""Stateless voice-cloning references for OpenAI-compatible speech endpoints.

OpenRouter's ``/audio/speech`` accepts ``input_references``: one to three reference
audio clips, each optionally followed by its transcript. The OpenAI SDK has no such
parameter, so callers of ``litellm.speech`` would otherwise lose it silently.

References come from one of two places, request first:

* the request body passes ``input_references`` in OpenRouter's shape, or
* the deployment binds a stored clip with ``reference_audio_path`` (and optionally
  ``reference_text``) in its ``litellm_params``, so each cloned voice is its own
  model alias and clients only pick a model name.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

MAX_REFERENCE_AUDIO_BYTES: Final = 15 * 1024 * 1024

_AUDIO_MIME_TYPES: Final = MappingProxyType(
    {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "opus": "audio/ogg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "webm": "audio/webm",
    }
)

_REFERENCE_LIST: Final = TypeAdapter(list[object])
_BODY: Final = TypeAdapter(Mapping[str, object])


class _AudioData(TypedDict):
    data: ReadOnly[str]
    format: ReadOnly[str]


class _AudioPart(TypedDict):
    type: ReadOnly[Literal["input_audio"]]
    input_audio: ReadOnly[_AudioData]


class _TextPart(TypedDict):
    type: ReadOnly[Literal["text"]]
    text: ReadOnly[str]


class SpeechReferenceBody(TypedDict):
    input_references: ReadOnly[Sequence[object]]


def resolve_speech_input_references(kwargs: Mapping[str, object]) -> Sequence[object] | None:
    """Return the ``input_references`` to forward, or ``None`` when there are none."""
    explicit: Final = kwargs.get("input_references")
    if isinstance(explicit, list) and explicit:
        return _REFERENCE_LIST.validate_python(explicit)

    path: Final = kwargs.get("reference_audio_path")
    if not isinstance(path, str) or not path.strip():
        return None
    return _REFERENCE_LIST.validate_python(
        _stored_clip_parts(Path(path.strip()).expanduser(), kwargs.get("reference_text"))
    )


def _stored_clip_parts(path: Path, transcript: object) -> tuple[_AudioPart | _TextPart, ...]:
    audio: Final = path.read_bytes()
    if not audio or len(audio) > MAX_REFERENCE_AUDIO_BYTES:
        raise ValueError(f"reference audio {path.name} must be 1 byte to {MAX_REFERENCE_AUDIO_BYTES} bytes")

    audio_format: Final = path.suffix.lstrip(".").lower() or "wav"
    mime: Final = _AUDIO_MIME_TYPES.get(audio_format, f"audio/{audio_format}")
    audio_part: Final[_AudioPart] = {
        "type": "input_audio",
        "input_audio": {
            "data": f"data:{mime};base64,{base64.b64encode(audio).decode('ascii')}",
            "format": audio_format,
        },
    }
    if isinstance(transcript, str) and transcript.strip():
        text_part: Final[_TextPart] = {"type": "text", "text": transcript.strip()}
        return audio_part, text_part
    return (audio_part,)


def redact_input_references(extra_body: object) -> object:
    """Telemetry view of ``extra_body``: reference audio is summarized, never logged."""
    if not isinstance(extra_body, Mapping) or "input_references" not in extra_body:
        return extra_body
    body: Final = _BODY.validate_python(extra_body)
    references: Final = body["input_references"]
    count: Final = len(_REFERENCE_LIST.validate_python(references)) if isinstance(references, list) else 0
    return _BODY.validate_python(MappingProxyType({**body, "input_references": f"[{count} reference part(s) omitted]"}))
