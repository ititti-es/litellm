from typing import Final

import pytest

from litellm.sessions.session import SEARCH_TEXT_LIMIT, Item, comparable, search_text


def _forms(items):
    return [form for _, form in comparable(items)]


def test_two_renderings_of_one_exchange_compare_equal():
    """What the gateway stored came from a provider; what a harness resends came from its own state."""
    stored = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        },
    ]
    resent = [
        {"role": "user", "content": "hi", "type": "message"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
    ]

    assert _forms(stored) == _forms(resent)


def test_different_words_do_not_compare_equal():
    said = [{"type": "message", "role": "user", "content": "hi"}]
    other = [{"type": "message", "role": "user", "content": "bye"}]

    assert _forms(said) != _forms(other)


def test_the_speaker_is_part_of_what_was_said():
    assert _forms([{"role": "user", "content": "hi"}]) != _forms([{"role": "assistant", "content": "hi"}])


def test_reasoning_is_left_out_so_a_harness_need_not_echo_it():
    """It is dropped on replay unless sealed, so its presence cannot decide whether two histories match."""
    with_thought = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
    ]

    assert _forms(with_thought) == _forms([{"type": "message", "role": "user", "content": "hi"}])


def test_positions_survive_the_items_that_were_left_out():
    """The caller maps a match back to its own list, which still contains what the comparison skipped."""
    items = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "role": "assistant", "content": "hello"},
    ]

    assert [index for index, _ in comparable(items)] == [0, 2]


def test_empty_assistant_output_does_not_change_tool_history_identity():
    call: Final[Item] = {"type": "function_call", "call_id": "c1", "name": "probe", "arguments": "{}"}
    assert _forms([{"role": "assistant", "content": []}, call]) == _forms([call])
    assert _forms([{"role": "assistant", "content": [{"type": "input_image"}]}, call]) != _forms([call])


def test_a_tool_call_is_compared_on_what_it_asked_for():
    call = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/"}'}]
    same = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/"}', "id": "fc_1"}]
    elsewhere = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/etc"}'}]

    assert _forms(call) == _forms(same)
    assert _forms(call) != _forms(elsewhere)


def test_a_tool_call_re_encoded_by_the_harness_still_compares_equal():
    """xAI sent 50.0 and opencode resent 50; the conversation forked into a full copy on that alone."""
    recorded = [
        {"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"limit": 50.0, "path": "/a"}'}
    ]
    resent = [{"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"path":"/a","limit":50}'}]
    wrapped = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "read",
            "arguments": '"{\\"path\\":\\"/a\\",\\"limit\\":50}"',
        }
    ]
    other = [{"type": "function_call", "call_id": "c1", "name": "read", "arguments": '{"path":"/a","limit":51}'}]
    text = [{"type": "function_call", "call_id": "c1", "name": "read", "arguments": "not json"}]

    assert _forms(recorded) == _forms(resent) == _forms(wrapped)
    assert _forms(recorded) != _forms(other)
    assert _forms(text) == [("function_call", "c1", "read", "not json")]


def test_a_tool_result_is_compared_on_what_it_returned():
    output = [{"type": "function_call_output", "call_id": "c1", "output": "a b"}]

    assert _forms(output) == _forms([{"type": "function_call_output", "call_id": "c1", "output": "a b"}])
    assert _forms(output) != _forms([{"type": "function_call_output", "call_id": "c1", "output": "c d"}])


def test_a_part_carrying_no_text_is_named_rather_than_ignored():
    """Two messages differing only by an attached image are not the same message."""
    with_image = [{"role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image"}]}]
    without = [{"role": "user", "content": [{"type": "input_text", "text": "look"}]}]

    assert _forms(with_image) != _forms(without)


@pytest.mark.parametrize(
    "item",
    [
        {"type": "reasoning", "content": "hidden", "summary": [{"type": "summary_text", "text": "hidden"}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "synthetic-image"}]},
        {"role": "user", "content": [{"type": "input_audio", "data": "synthetic-audio"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hidden"}]},
        {"role": "assistant", "content": ""},
    ],
)
def test_search_text_excludes_reasoning_nontext_and_empty_items(item: Item) -> None:
    assert search_text(item) is None


def test_search_text_keeps_only_text_from_mixed_message_parts() -> None:
    item: Final[Item] = {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "first"},
            {"type": "input_image", "image_url": "synthetic-image"},
            {"type": "input_audio", "data": "synthetic-audio"},
            {"type": "thinking", "thinking": "hidden"},
            {"type": "text", "text": "second"},
        ],
    }

    assert search_text(item) == "first\nsecond"


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"role": "user", "content": "plain text"}, "plain text"),
        ({"type": "function_call", "name": "lookup", "arguments": '{"query":"synthetic"}'}, '{"query":"synthetic"}'),
        ({"type": "function_call_output", "call_id": "call-1", "output": "synthetic result"}, "synthetic result"),
        (
            {
                "type": "function_call_output",
                "output": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
            },
            "first\nsecond",
        ),
    ],
)
def test_search_text_indexes_message_text_tool_arguments_and_tool_results(item: Item, expected: str) -> None:
    assert search_text(item) == expected


def test_search_text_is_capped_so_a_huge_output_still_indexes():
    huge: Item = {"type": "function_call_output", "call_id": "c", "output": "word " * SEARCH_TEXT_LIMIT}
    text = search_text(huge)
    assert text is not None and len(text) == SEARCH_TEXT_LIMIT
