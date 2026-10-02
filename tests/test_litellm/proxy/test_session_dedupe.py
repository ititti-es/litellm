from litellm.proxy.session_dedupe import Row, form_digests, redundant_rows


def _user(text: str):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _reply(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _row(session_id: str, *items) -> Row:
    return Row(session_id, len(items), form_digests(items))


def test_each_copy_a_longer_row_reproduces_is_redundant_and_the_longest_is_kept():
    one = (_user("a"), _reply("b"))
    two = (*one, _user("c"), _reply("d"))
    three = (*two, _user("e"))
    rows = (_row("first", *one), _row("second", *two), _row("third", *three))

    assert redundant_rows(rows) == {"first": "third", "second": "third"}


def test_separate_conversations_under_one_id_are_all_kept():
    main = (_user("main"), _reply("ok"), _user("more"))
    subagent = (_user("subagent"), _reply("ok"))
    title = (_user("main"), _reply("A title"))

    assert redundant_rows((_row("main", *main), _row("subagent", *subagent), _row("title", *title))) == {}


def test_a_fork_keeps_both_branches_and_drops_only_their_shared_copy():
    shared = (_user("a"), _reply("b"))
    rows = (
        _row("shared", *shared),
        _row("left", *shared, _user("left")),
        _row("right", *shared, _user("right")),
    )

    covered = redundant_rows(rows)

    assert set(covered) == {"shared"} and covered["shared"] in {"left", "right"}


def test_reasoning_and_rendering_differences_do_not_keep_a_copy_alive():
    """Compared as the gateway compares a resend: reasoning is left out and a string equals its text part."""
    stored = (_user("a"), {"type": "reasoning", "id": "rs_1", "summary": []}, _reply("b"))
    resent = ({"type": "message", "role": "user", "content": "a"}, _reply("b"), _user("c"))

    assert redundant_rows((_row("stored", *stored), _row("resent", *resent))) == {"stored": "resent"}


def test_identical_copies_keep_exactly_one():
    items = (_user("a"), _reply("b"))

    covered = redundant_rows((_row("x", *items), _row("y", *items)))

    assert len(covered) == 1
