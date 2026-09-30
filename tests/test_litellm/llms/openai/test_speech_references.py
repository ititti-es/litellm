import base64

import pytest

from litellm.litellm_core_utils.logging_utils import speech_request_body
from litellm.llms.openai.speech_references import (
    MAX_REFERENCE_AUDIO_BYTES,
    redact_input_references,
    resolve_speech_input_references,
)


def test_no_references_returns_none():
    assert resolve_speech_input_references({}) is None
    assert resolve_speech_input_references({"input_references": []}) is None
    assert resolve_speech_input_references({"reference_audio_path": "  "}) is None


def test_request_references_win_over_stored_clip(tmp_path):
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    explicit = [{"type": "text", "text": "hi"}]

    resolved = resolve_speech_input_references({"input_references": explicit, "reference_audio_path": str(clip)})

    assert resolved == explicit


def test_stored_clip_becomes_audio_and_transcript_parts(tmp_path):
    clip = tmp_path / "david.wav"
    clip.write_bytes(b"RIFFdata")

    parts = resolve_speech_input_references({"reference_audio_path": str(clip), "reference_text": " Hello there. "})

    assert parts == [
        {
            "type": "input_audio",
            "input_audio": {
                "data": "data:audio/wav;base64," + base64.b64encode(b"RIFFdata").decode(),
                "format": "wav",
            },
        },
        {"type": "text", "text": "Hello there."},
    ]


def test_stored_clip_without_transcript_has_only_audio(tmp_path):
    clip = tmp_path / "voice.mp3"
    clip.write_bytes(b"ID3")

    parts = resolve_speech_input_references({"reference_audio_path": str(clip)})

    assert parts is not None and len(parts) == 1
    assert parts[0]["input_audio"]["format"] == "mp3"


@pytest.mark.parametrize("size", [0, MAX_REFERENCE_AUDIO_BYTES + 1])
def test_stored_clip_size_is_bounded(tmp_path, size):
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"\0" * size)

    with pytest.raises(ValueError, match=r"reference audio .* must be"):
        resolve_speech_input_references({"reference_audio_path": str(clip)})


def test_telemetry_body_never_carries_reference_audio():
    extra_body = {"input_references": [{"type": "input_audio", "input_audio": {"data": "AAAA"}}], "other": 1}

    body = speech_request_body("m", "alloy", {"response_format": "mp3", "extra_body": extra_body})

    assert body["extra_body"] == {"input_references": "[1 reference part(s) omitted]", "other": 1}
    assert "AAAA" not in repr(body)
    assert redact_input_references({"other": 1}) == {"other": 1}


def test_speech_sends_stored_clip_as_extra_body_and_drops_voice(tmp_path, monkeypatch):
    import openai

    import litellm
    import litellm.main as litellm_main

    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF")
    captured = {}

    def fake_audio_speech(**kwargs):
        captured.update(kwargs)
        return "audio"

    monkeypatch.setattr(litellm_main.openai_chat_completions, "audio_speech", fake_audio_speech)

    litellm.speech(
        model="openai/fish-audio/s2.1-pro-free:free",
        input="hi",
        voice="alloy",
        api_key="sk-test",
        api_base="https://openrouter.ai/api/v1",
        reference_audio_path=str(clip),
    )

    assert captured["voice"] is openai.NOT_GIVEN
    assert captured["optional_params"]["extra_body"]["input_references"][0]["type"] == "input_audio"


def test_speech_without_references_keeps_voice(monkeypatch):
    import litellm
    import litellm.main as litellm_main

    captured = {}

    def fake_audio_speech(**kwargs):
        captured.update(kwargs)
        return "audio"

    monkeypatch.setattr(litellm_main.openai_chat_completions, "audio_speech", fake_audio_speech)

    litellm.speech(model="openai/tts-1", input="hi", voice="alloy", api_key="sk-test")

    assert captured["voice"] == "alloy"
    assert "extra_body" not in captured["optional_params"]
