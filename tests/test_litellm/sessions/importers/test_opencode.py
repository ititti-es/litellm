import json
import sqlite3

from litellm.sessions.importers.opencode import read_opencode


def _db(path, sessions, messages, parts):
    db = sqlite3.connect(path)
    db.executescript(
        "create table session (id text, parent_id text, title text, version text, revert text, time_created int,"
        " time_updated int, directory text);"
        "create table message (id text, session_id text, time_created int, time_updated int, data text);"
        "create table part (id text, message_id text, session_id text, time_created int, time_updated int, data text);"
    )
    db.executemany("insert into session values (?,?,?,?,?,?,?,?)", sessions)
    db.executemany("insert into message values (?,?,?,?,?)", [(m, s, t, t, json.dumps(d)) for m, s, t, d in messages])
    db.executemany("insert into part values (?,?,?,?,?,?)", [(p, m, s, 0, 0, json.dumps(d)) for p, m, s, d in parts])
    db.commit()
    db.close()


ASSISTANT = {"role": "assistant", "providerID": "xai", "modelID": "grok-4", "agent": "build"}


def _sessions(tmp_path, revert=None):
    path = tmp_path / "opencode.db"
    _db(
        path,
        [
            ("ses_parent", None, "Parent", "1.2.3", json.dumps(revert) if revert else None, 1, 9, "/w"),
            ("ses_child", "ses_parent", "Child", "1.2.3", None, 2, 3, "/w"),
        ],
        [
            ("msg_1", "ses_parent", 1, {"role": "user"}),
            ("msg_2", "ses_parent", 2, ASSISTANT),
            ("msg_3", "ses_parent", 3, {**ASSISTANT, "error": {"name": "APIError"}}),
            ("msg_4", "ses_parent", 4, {"role": "user"}),
            ("msg_5", "ses_parent", 5, {**ASSISTANT, "summary": True}),
            ("msg_6", "ses_child", 6, {"role": "user"}),
        ],
        [
            ("prt_1", "msg_1", "ses_parent", {"type": "text", "text": "list files"}),
            ("prt_2", "msg_2", "ses_parent", {"type": "step-start"}),
            ("prt_3", "msg_2", "ses_parent", {"type": "reasoning", "text": "thinking"}),
            (
                "prt_4",
                "msg_2",
                "ses_parent",
                {
                    "type": "tool",
                    "callID": "c1",
                    "tool": "bash",
                    "state": {
                        "status": "completed",
                        "input": {"command": "ls"},
                        "output": "a b",
                        "time": {"compacted": 5},
                    },
                },
            ),
            (
                "prt_5",
                "msg_2",
                "ses_parent",
                {"type": "tool", "callID": "c2", "tool": "read", "state": {"status": "running", "input": {}}},
            ),
            ("prt_6", "msg_2", "ses_parent", {"type": "text", "text": "done"}),
            ("prt_7", "msg_3", "ses_parent", {"type": "text", "text": "never sent"}),
            ("prt_8", "msg_4", "ses_parent", {"type": "compaction", "auto": True}),
            ("prt_9", "msg_5", "ses_parent", {"type": "text", "text": "summary of work"}),
            ("prt_a", "msg_6", "ses_child", {"type": "text", "text": "subtask prompt"}),
        ],
    )
    return {s.source_id: s for s in read_opencode(path)}


def test_a_session_becomes_neutral_items_with_provenance_and_harness_marks(tmp_path):
    parent = _sessions(tmp_path)["ses_parent"]
    kinds = [item.meta["kind"] for item in parent.items]
    assert kinds == [
        "user",
        "reasoning",
        "tool_call",
        "tool_output",
        "tool_call",
        "interrupted_tool",
        "text",
        "compaction_request",
        "compaction_summary",
    ]
    call, output = parent.items[2], parent.items[3]
    assert call.item == {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": '{"command": "ls"}'}
    assert call.meta["tool"] == {"name": "bash", "class": "shell"}
    assert call.meta["provenance"]["model"] == "grok-4" and call.meta["provenance"]["harness"] == "opencode"
    assert output.item["output"] == "a b" and output.meta["pruned_by_harness"] is True
    assert parent.items[5].item["call_id"] == "c2"
    assert parent.turn_ends == (7, 9)
    assert parent.skipped == 0


def test_children_link_to_their_parent_and_reverted_history_is_dropped(tmp_path):
    sessions = _sessions(tmp_path, revert={"messageID": "msg_4"})
    assert sessions["ses_child"].parent_source_id == "ses_parent"
    assert [item.meta["kind"] for item in sessions["ses_parent"].items][-1] == "text"


def _v2_db(path, messages):
    db = sqlite3.connect(path)
    db.executescript(
        "create table session_v2 (id text, parent_id text, title text, version text, revert text, time_created int,"
        " time_updated int, directory text);"
        "create table session_message (id text, session_id text, type text, seq int, time_created int,"
        " time_updated int, data text);"
    )
    db.execute("insert into session_v2 values ('ses_v2', 'ses_root', 'V2', '2.0', null, 1, 2, '/w')")
    db.executemany(
        "insert into session_message values (?, 'ses_v2', ?, ?, 0, 0, ?)",
        [(f"msg_{seq}", kind, seq, json.dumps(data)) for seq, (kind, data) in enumerate(messages)],
    )
    db.commit()
    db.close()


def test_the_v2_store_format_is_read_with_types_from_their_column(tmp_path):
    model = {"id": "gpt-6-astra", "providerID": "openai"}
    _v2_db(
        tmp_path / "v2.db",
        [
            ("user", {"text": "fix it", "files": [{"uri": "data:image/png;base64,AA", "mime": "image/png"}]}),
            ("model-switched", {"model": model}),
            (
                "assistant",
                {
                    "agent": "build",
                    "model": model,
                    "providerState": {"responseId": "resp_up"},
                    "content": [
                        {"type": "reasoning", "id": "r", "text": "hmm"},
                        {
                            "type": "tool",
                            "id": "call_1",
                            "name": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": "ls"},
                                "content": [{"type": "text", "text": "a"}],
                            },
                            "time": {"pruned": 9},
                        },
                        {
                            "type": "tool",
                            "id": "call_2",
                            "name": "read",
                            "state": {"status": "error", "input": {}, "error": {"message": "nope"}},
                        },
                        {"type": "text", "id": "t", "text": "done"},
                    ],
                },
            ),
            ("assistant", {"model": model, "content": [], "error": {"type": "unknown", "message": "boom"}}),
            ("system", {"text": "operator note"}),
            ("compaction", {"summary": "so far", "recent": "", "reason": "auto"}),
            ("some-future-type", {}),
        ],
    )
    (session,) = read_opencode(tmp_path / "v2.db")
    assert [item.meta["kind"] for item in session.items] == [
        "user",
        "reasoning",
        "tool_call",
        "tool_output",
        "tool_call",
        "tool_output",
        "text",
        "system_note",
        "compaction_summary",
    ]
    assert session.items[0].item["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,AA"}
    assert session.items[3].item["output"] == "a" and session.items[3].meta["pruned_by_harness"] is True
    assert session.items[5].item["output"] == "Error: nope"
    assert session.items[2].meta["provenance"]["provider_response_id"] == "resp_up"
    assert session.parent_source_id == "ses_root" and session.metadata["format"] == "v2"
    assert session.turn_ends == (7,) and session.skipped == 1
