"""
Regression tests for LIT-4210: the streaming iterators must never run the sync
success_handler on the thread-pool executor concurrently with
async_success_handler. Concurrent mutation of the shared response object /
model_call_details from two threads segfaults pydantic-core (customer pods
crashed with exit 139 whenever any CustomLogger was registered).
"""

import asyncio
import time

import httpx
import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils import thread_pool_executor as thread_pool_executor_module
from litellm.responses import streaming_iterator as responses_streaming_iterator_module
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.responses.streaming_iterator import ResponsesAPIStreamingIterator
from litellm.types.llms.openai import ResponsesAPIResponse


class RecordingCustomLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.async_hook_started: float | None = None
        self.async_hook_finished: float | None = None

    async def _record(self):
        self.async_hook_started = time.monotonic()
        await asyncio.sleep(0.2)
        self.async_hook_finished = time.monotonic()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._record()

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        await self._record()


class RecordingExecutor:
    def __init__(self, inner):
        self._inner = inner
        self.submits: list = []

    def submit(self, fn, *args, **kwargs):
        self.submits.append((time.monotonic(), fn))
        return self._inner.submit(fn, *args, **kwargs)

    def submit_times_for(self, logging_obj) -> list:
        return [t for t, fn in self.submits if getattr(fn, "__self__", None) is logging_obj]


@pytest.fixture(autouse=True)
def _isolate_callbacks():
    saved = (
        litellm.callbacks,
        litellm.success_callback,
        litellm._async_success_callback,
        litellm.failure_callback,
        litellm._async_failure_callback,
    )
    yield
    (
        litellm.callbacks,
        litellm.success_callback,
        litellm._async_success_callback,
        litellm.failure_callback,
        litellm._async_failure_callback,
    ) = saved


@pytest.fixture
def recording_executor(monkeypatch):
    recording = RecordingExecutor(thread_pool_executor_module.executor)
    monkeypatch.setattr(thread_pool_executor_module, "executor", recording)
    monkeypatch.setattr(responses_streaming_iterator_module, "executor", recording)
    return recording


def _make_logging_obj() -> LitellmLogging:
    logging_obj = LitellmLogging(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="aresponses",
        start_time=time.time(),
        litellm_call_id="lit-4210-test",
        function_id="lit-4210-test",
    )
    logging_obj.model_call_details["litellm_params"] = {"aresponses": True}
    return logging_obj


def _make_iterator(logging_obj: LitellmLogging) -> ResponsesAPIStreamingIterator:
    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=None,
        logging_obj=logging_obj,
    )
    iterator.completed_response = ResponsesAPIResponse(
        id="resp_lit4210",
        created_at=1700000000.0,
        model="gpt-5.4-nano",
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata={},
        temperature=1.0,
        top_p=1.0,
    )
    return iterator


@pytest.mark.asyncio
async def test_custom_logger_only_never_submits_sync_success_handler(recording_executor):
    recorder = RecordingCustomLogger()
    litellm.success_callback = [recorder]
    litellm._async_success_callback = [recorder]

    logging_obj = _make_logging_obj()
    iterator = _make_iterator(logging_obj)

    iterator._log_completed_response(is_async=True)
    await asyncio.sleep(0.6)

    assert recorder.async_hook_started is not None
    assert recording_executor.submit_times_for(logging_obj) == []


@pytest.mark.asyncio
async def test_sync_callbacks_run_only_after_async_handler_completes(recording_executor):
    recorder = RecordingCustomLogger()
    sync_events: list = []

    def sync_callback(kwargs, response_obj, start_time, end_time):
        sync_events.append(time.monotonic())

    litellm.success_callback = [recorder, sync_callback]
    litellm._async_success_callback = [recorder]

    logging_obj = _make_logging_obj()
    iterator = _make_iterator(logging_obj)

    iterator._log_completed_response(is_async=True)
    await asyncio.sleep(0.8)

    assert recorder.async_hook_finished is not None
    submit_times = recording_executor.submit_times_for(logging_obj)
    assert len(submit_times) == 1
    assert submit_times[0] >= recorder.async_hook_finished


@pytest.mark.parametrize(("event", "notified"), [("response.completed", True), ("response.failed", False)])
async def test_on_completed_fires_only_for_a_completed_response(event, notified):
    import json

    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    received: list = []
    iterator.on_completed = received.append
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
    iterator._process_chunk(
        json.dumps({"type": event, "response": {"id": "resp_x", "created_at": 0, "status": "completed", "output": output}})
    )

    from litellm.sessions.responses import provider_response_id

    assert [(provider_response_id(r.id), r.output) for r in received] == ([("resp_x", output)] if notified else [])


async def test_the_completed_event_is_held_until_the_async_completion_hook_finishes():
    import json
    from types import SimpleNamespace

    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    output = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
    completed = {"type": "response.completed", "response": {"id": "resp_x", "created_at": 0, "status": "completed", "output": output}}

    async def events():
        yield SimpleNamespace(data=json.dumps(completed))

    iterator.stream_iterator = events()
    order: list = []
    sync_calls: list = []

    async def stored(response):
        await asyncio.sleep(0.05)
        order.append("stored")

    iterator.on_completed = sync_calls.append
    iterator.on_completed_async = stored

    chunk = await iterator.__anext__()
    order.append(chunk.type)

    assert order == ["stored", "response.completed"]
    assert sync_calls == []


def _item_fields(item, *keys):
    raw = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
    return tuple(raw.get(key) for key in keys)


async def test_a_completed_event_without_output_hands_the_hook_the_streamed_items():
    """ChatGPT's Codex backend ends `store: false` streams with `output: []`; the session store must still see the reply."""
    import json

    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    received: list = []
    iterator.on_completed = received.append
    call = {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "ls", "arguments": "{}", "status": "completed"}
    text = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "looking", "annotations": []}],
    }
    iterator._process_chunk(json.dumps({"type": "response.output_item.done", "output_index": 1, "item": call}))
    iterator._process_chunk(json.dumps({"type": "response.output_item.done", "output_index": 0, "item": text}))
    terminal = iterator._process_chunk(
        json.dumps({"type": "response.completed", "response": {"id": "resp_x", "created_at": 0, "status": "completed", "output": []}})
    )

    assert len(received) == 1
    assert [_item_fields(item, "type", "call_id") for item in received[0].output] == [("message", None), ("function_call", "c1")]
    assert terminal.response.output == []


async def test_a_completed_event_with_output_is_handed_on_as_it_is():
    import json

    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig

    iterator = ResponsesAPIStreamingIterator(
        response=httpx.Response(200),
        model="gpt-5.4-nano",
        responses_api_provider_config=OpenAIResponsesAPIConfig(),
        logging_obj=_make_logging_obj(),
    )
    received: list = []
    iterator.on_completed = received.append
    streamed = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": []}
    final = [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}]
    iterator._process_chunk(json.dumps({"type": "response.output_item.done", "output_index": 0, "item": streamed}))
    iterator._process_chunk(
        json.dumps({"type": "response.completed", "response": {"id": "resp_x", "created_at": 0, "status": "completed", "output": final}})
    )

    assert [r.output for r in received] == [final]
