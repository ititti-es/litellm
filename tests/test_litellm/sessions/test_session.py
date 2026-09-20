from litellm.sessions.session import comparable


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


def test_a_tool_call_is_compared_on_what_it_asked_for():
    call = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/"}'}]
    same = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/"}', "id": "fc_1"}]
    elsewhere = [{"type": "function_call", "call_id": "c1", "name": "ls", "arguments": '{"path": "/etc"}'}]

    assert _forms(call) == _forms(same)
    assert _forms(call) != _forms(elsewhere)


def test_a_tool_result_is_compared_on_what_it_returned():
    output = [{"type": "function_call_output", "call_id": "c1", "output": "a b"}]

    assert _forms(output) == _forms([{"type": "function_call_output", "call_id": "c1", "output": "a b"}])
    assert _forms(output) != _forms([{"type": "function_call_output", "call_id": "c1", "output": "c d"}])


def test_a_part_carrying_no_text_is_named_rather_than_ignored():
    """Two messages differing only by an attached image are not the same message."""
    with_image = [{"role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image"}]}]
    without = [{"role": "user", "content": [{"type": "input_text", "text": "look"}]}]

    assert _forms(with_image) != _forms(without)
