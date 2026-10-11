"""Speech bodies are relayed as the provider sends them, and always closed."""

from unittest.mock import MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from litellm.llms.openai.openai import OpenAIChatCompletion
from litellm.proxy.proxy_server import _audio_speech_chunk_generator


class _Body(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        yield b"ID3-first"
        yield b"-rest"

    async def aclose(self) -> None:
        self.closed = True


async def _speech(body: _Body, status: int = 200):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, headers={"content-type": "audio/mpeg"}, stream=body)
    )
    client = AsyncOpenAI(
        api_key="k", base_url="https://upstream.test/v1", http_client=httpx.AsyncClient(transport=transport)
    )
    return await OpenAIChatCompletion().async_audio_speech(
        model="m",
        input="hi",
        voice="alloy",
        optional_params={},
        api_key="k",
        api_base=None,
        organization=None,
        project=None,
        max_retries=0,
        timeout=10,
        logging_obj=MagicMock(),
        client=client,
    )


@pytest.mark.asyncio
async def test_speech_body_is_not_read_before_relay() -> None:
    body = _Body()
    response = await _speech(body)
    assert not response.response.is_stream_consumed
    chunks = [c async for c in _audio_speech_chunk_generator(response)]
    assert b"".join(chunks) == b"ID3-first-rest"
    assert body.closed


@pytest.mark.asyncio
async def test_disconnect_mid_clip_closes_upstream() -> None:
    body = _Body()
    generator = _audio_speech_chunk_generator(await _speech(body))
    await generator.__anext__()
    await generator.aclose()
    assert body.closed


@pytest.mark.asyncio
async def test_status_error_raises_before_relay() -> None:
    import openai

    with pytest.raises(openai.APIStatusError):
        await _speech(_Body(), status=400)
