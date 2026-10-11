import json

from litellm.sessions.importers.claude_code import (
    read_claude_code,
    read_claude_code_session,
    transcript_path,
)

SESSION = "11111111-2222-3333-4444-555555555555"


def _write(tmp_path, records, session=SESSION, project="-home-sett-work"):
    directory = tmp_path / project
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _record(uuid, parent, **fields):
    return {
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "timestamp": "2026-09-20T10:00:00.000Z",
        "version": "2.1.278",
        "cwd": "/home/user/work",
        "isSidechain": False,
        **fields,
    }


def _user(uuid, parent, content, **fields):
    return _record(uuid, parent, type="user", message={"role": "user", "content": content}, **fields)


def _assistant(uuid, parent, content, message_id="msg_1", **message):
    return _record(
        uuid,
        parent,
        type="assistant",
        message={"id": message_id, "role": "assistant", "model": "claude-opus-5", "content": content, **message},
    )


def _kinds(session):
    return [item.meta["kind"] for item in session.items]


# --- the shape of an imported conversation ----------------------------------------------------------


def test_a_conversation_becomes_neutral_items(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "read the file"),
            _assistant("a1", "u1", [{"type": "thinking", "thinking": "hmm", "signature": "sealed"}]),
            _assistant("a2", "a1", [{"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.py"}}]),
            _user("t1", "a2", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file contents"}]),
            _assistant("a3", "t1", [{"type": "text", "text": "it reads a file"}], message_id="msg_2"),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "reasoning", "tool_call", "tool_output", "text"]
    assert session.items[0].item == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "read the file"}],
    }
    assert session.items[2].item["call_id"] == "toolu_1"
    assert session.items[2].item["arguments"] == json.dumps({"path": "a.py"})
    assert session.items[3].item == {"type": "function_call_output", "call_id": "toolu_1", "output": "file contents"}
    assert session.source == "claude_code"
    assert session.source_id == SESSION


def test_model_output_is_marked_as_output(tmp_path):
    """Only the assistant's own items may be reduced for replay; a tool result is input the model was given."""
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]),
            _user("t1", "a1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "done"}]),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert [item.is_output for item in session.items] == [False, True, False]


def test_a_reply_is_one_turn_however_many_records_it_took(tmp_path):
    """Claude Code writes thinking, text and each tool call as separate records of one API message."""
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "thinking", "thinking": "plan", "signature": "s"}]),
            _assistant("a2", "a1", [{"type": "text", "text": "on it"}]),
            _assistant("a3", "a2", [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]),
            _user("t1", "a3", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}]),
            _assistant("a4", "t1", [{"type": "text", "text": "done"}], message_id="msg_2"),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    # Two replies: the three records of msg_1, then msg_2. Each end is after that reply's last item.
    assert session.turn_ends == (4, 6)


def test_thinking_is_bound_to_the_provider_that_signed_it(tmp_path):
    """Anthropic only accepts its own signature back, so a view for another provider has to know to drop it."""
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "thinking", "thinking": "why", "signature": "sig"}]),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)
    thought = session.items[1]

    assert thought.item["encrypted_content"] == "sig"
    assert thought.item["summary"] == [{"type": "summary_text", "text": "why"}]
    assert thought.meta["provider"] == "anthropic"


def test_an_unsigned_thought_is_not_bound_to_a_provider(tmp_path):
    _write(tmp_path, [_user("u1", None, "go"), _assistant("a1", "u1", [{"type": "thinking", "thinking": "why"}])])

    thought = read_claude_code_session(SESSION, tmp_path).items[1]

    assert "encrypted_content" not in thought.item
    assert "provider" not in thought.meta


def test_a_failed_tool_result_is_marked(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]),
            _user("t1", "a1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "boom", "is_error": True}]),
        ],
    )

    assert read_claude_code_session(SESSION, tmp_path).items[2].meta["is_error"] is True


def test_a_turn_with_tool_results_and_a_message_keeps_both(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]),
            _user(
                "t1",
                "a1",
                [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                    {"type": "text", "text": "and also do this"},
                ],
            ),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "tool_call", "tool_output", "user"]
    assert session.items[3].item["content"] == [{"type": "input_text", "text": "and also do this"}]


def test_provenance_records_what_produced_the_reply(tmp_path):
    _write(
        tmp_path,
        [_user("u1", None, "go"), _assistant("a1", "u1", [{"type": "text", "text": "hi"}], stop_reason="end_turn")],
    )

    provenance = read_claude_code_session(SESSION, tmp_path).items[1].meta["provenance"]

    assert provenance["harness"] == "claude_code"
    assert provenance["harness_version"] == "2.1.278"
    assert provenance["provider"] == "anthropic"
    assert provenance["model"] == "claude-opus-5"
    assert provenance["stop_reason"] == "end_turn"


def test_a_tool_call_carries_its_canonical_class(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}}]),
        ],
    )

    assert read_claude_code_session(SESSION, tmp_path).items[1].meta["tool"] == {"name": "Bash", "class": "shell"}


# --- branches and compaction -------------------------------------------------------------------------


def test_only_the_branch_the_harness_would_resume_is_imported(tmp_path):
    """A rewound conversation leaves the abandoned turns in the file; resuming it would not replay them."""
    _write(
        tmp_path,
        [
            _user("u1", None, "first"),
            _assistant("a1", "u1", [{"type": "text", "text": "abandoned"}]),
            _assistant("a2", "u1", [{"type": "text", "text": "kept"}], message_id="msg_2"),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert [item.item["content"][0]["text"] for item in session.items] == ["first", "kept"]


def test_compacted_turns_are_kept_with_the_summary_that_replaced_them(tmp_path):
    """The boundary record has no parent; the history it dropped hangs off logicalParentUuid."""
    _write(
        tmp_path,
        [
            _user("u1", None, "the long original conversation"),
            _assistant("a1", "u1", [{"type": "text", "text": "at length"}]),
            _record(
                "b1",
                None,
                logicalParentUuid="a1",
                type="system",
                subtype="compact_boundary",
                content="Conversation compacted",
                compactMetadata={"trigger": "manual", "preTokens": 300000},
            ),
            _user("s1", "b1", "This session is being continued...", isCompactSummary=True),
            _assistant("a2", "s1", [{"type": "text", "text": "carrying on"}], message_id="msg_2"),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "text", "compaction_boundary", "compaction_summary", "text"]
    assert session.items[2].meta["source"]["compaction"]["preTokens"] == 300000


def test_a_harness_note_is_stored_as_what_it_is(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _record(
                "e1",
                "u1",
                type="attachment",
                attachment={"type": "environment"},
                rendered=[{"content": "<system-reminder>cwd is /home/user/work</system-reminder>"}],
            ),
            _assistant("a1", "e1", [{"type": "text", "text": "noted"}]),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "harness_context", "text"]
    assert session.items[1].meta["source"]["attachment"] == "environment"


def test_per_turn_reminders_are_not_stored(tmp_path):
    """The harness rebuilds these from live state every turn; storing them would archive a token counter."""
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _record(
                "r1",
                "u1",
                type="attachment",
                attachment={"type": "total_tokens_reminder"},
                rendered=[{"content": "<system-reminder>14944578 tokens left</system-reminder>"}],
            ),
            _assistant("a1", "r1", [{"type": "text", "text": "ok"}]),
        ],
    )

    assert _kinds(read_claude_code_session(SESSION, tmp_path)) == ["user", "text"]


def test_turn_telemetry_is_not_content(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "text", "text": "ok"}]),
            _record("d1", "a1", type="system", subtype="turn_duration", durationMs=1200, content="1.2s"),
        ],
    )

    assert _kinds(read_claude_code_session(SESSION, tmp_path)) == ["user", "text"]


# --- reading the files -------------------------------------------------------------------------------


def test_a_half_written_final_line_is_counted_not_fatal(tmp_path):
    """The harness appends while the proxy reads; the last line can be incomplete."""
    path = _write(tmp_path, [_user("u1", None, "go"), _assistant("a1", "u1", [{"type": "text", "text": "ok"}])])
    with open(path, "a") as transcript:
        transcript.write('{"uuid": "a2", "parentUu')

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "text"]
    assert session.skipped == 1


def test_a_subagent_is_left_out_of_the_parent_and_counted(tmp_path):
    _write(
        tmp_path,
        [
            _user("u1", None, "go"),
            _assistant("a1", "u1", [{"type": "tool_use", "id": "toolu_1", "name": "Task", "input": {}}]),
            {
                **_record("s1", "a1", type="user", message={"role": "user", "content": "subagent work"}),
                "isSidechain": True,
            },
            _user("t1", "a1", [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"}]),
        ],
    )

    session = read_claude_code_session(SESSION, tmp_path)

    assert _kinds(session) == ["user", "tool_call", "tool_output"]
    assert session.skipped == 1
    assert session.metadata["sidechain_records"] == 1


def test_every_transcript_under_a_projects_directory_is_read(tmp_path):
    _write(tmp_path, [_user("u1", None, "one")], session="aaaa", project="-home-sett-a")
    _write(tmp_path, [_user("u1", None, "two")], session="bbbb", project="-home-sett-b")

    assert {session.source_id for session in read_claude_code(tmp_path)} == {"aaaa", "bbbb"}


def test_a_session_with_no_transcript_is_not_an_error(tmp_path):
    assert read_claude_code_session("no-such-session", tmp_path) is None
    assert transcript_path("no-such-session", tmp_path) is None


def test_an_empty_transcript_yields_nothing(tmp_path):
    _write(tmp_path, [])

    assert list(read_claude_code(tmp_path)) == []


def test_the_session_is_named_by_the_harness(tmp_path):
    _write(tmp_path, [_user("u1", None, "go"), {"type": "ai-title", "aiTitle": "Fixing the transcript gap"}])

    assert read_claude_code_session(SESSION, tmp_path).title == "Fixing the transcript gap"


def test_where_it_ran_is_recorded(tmp_path):
    _write(tmp_path, [_user("u1", None, "go"), _assistant("a1", "u1", [{"type": "text", "text": "ok"}])])

    metadata = read_claude_code_session(SESSION, tmp_path).metadata

    assert metadata["harness"] == "claude_code"
    assert metadata["cwd"] == "/home/user/work"
    assert metadata["model"] == "claude-opus-5"


def test_reading_a_copy_leaves_the_harnesss_file_alone(tmp_path):
    path = _write(tmp_path, [_user("u1", None, "go")])
    before = path.read_bytes()
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    session = read_claude_code_session(SESSION, tmp_path, scratch)

    assert _kinds(session) == ["user"]
    assert path.read_bytes() == before
    assert list(scratch.iterdir()) == []
