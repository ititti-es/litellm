"""Whether a caller who did not ask to stream is answered with one, for a provider that always streams upstream"""

import pytest

import litellm
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.sessions.responses import provider_response_id


@pytest.fixture
def chatgpt():
    return ChatGPTResponsesAPIConfig()


def test_the_codex_backend_only_answers_with_an_event_stream(chatgpt):
    assert chatgpt.forces_upstream_stream is True


def test_a_caller_who_did_not_ask_to_stream_is_not_handed_a_stream(chatgpt):
    """The provider's own request says stream: true, but that describes the wire, not the caller."""
    assert chatgpt.is_stream_request(False, {"stream": True}, None) is False


def test_a_caller_who_asked_to_stream_still_gets_one(chatgpt):
    assert chatgpt.is_stream_request(True, {"stream": True}, None) is True


def test_an_extra_body_can_still_ask_for_a_stream(chatgpt):
    assert chatgpt.is_stream_request(False, {"stream": True}, {"stream": True}) is True


def test_other_providers_keep_treating_the_request_stream_flag_as_the_callers():
    """Only a provider that declares it forces streaming is affected."""
    openai = OpenAIResponsesAPIConfig()

    assert openai.forces_upstream_stream is False
    assert openai.is_stream_request(False, {"stream": True}, None) is True
    assert openai.is_stream_request(False, {}, None) is False


async def test_a_non_streaming_call_returns_one_response_not_an_iterator(monkeypatch):
    """End to end through the handler, with the backend replaced by the event stream it really sends."""
    import httpx

    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    sse = (
        'data: {"type":"response.created","response":{"id":"resp_1","object":"response","status":"in_progress",'
        '"output":[]}}\n\n'
        'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"m1",'
        '"role":"assistant","status":"completed","content":[{"type":"output_text","text":"ready","annotations":[]}]}}\n\n'
        'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed",'
        '"created_at":0,"model":"gpt-5.6-sol","output":[],"usage":{"input_tokens":3,"output_tokens":1,"total_tokens":4}}}\n\n'
    )

    async def post(self, url, **kwargs):
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(AsyncHTTPHandler, "post", post)
    monkeypatch.setattr(
        "litellm.llms.chatgpt.responses.transformation.ChatGPTResponsesAPIConfig.validate_environment",
        lambda self, headers, model, litellm_params=None: {"accept": "text/event-stream"},
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.responses.transformation.ChatGPTResponsesAPIConfig.get_complete_url",
        lambda self, api_base, litellm_params: "https://chatgpt.example/backend-api/codex/responses",
    )

    response = await litellm.aresponses(model="chatgpt/gpt-5.6-sol", input="hi", stream=False)

    assert not hasattr(response, "__aiter__")
    assert provider_response_id(response.id) == "resp_1"  # LiteLLM wraps the provider id in its own envelope
    assert [part.text for item in response.output for part in item.content] == ["ready"]
