import json
import sqlite3

from litellm.sessions.importers.openclaw import read_openclaw


def _event(event_id, parent, **fields):
    return {"id": event_id, "parentId": parent, "timestamp": "2026-09-19T00:00:00Z", **fields}


def _message(event_id, parent, **message):
    return _event(event_id, parent, type="message", message=message)


def _state(tmp_path, events, leaf):
    agent_dir = tmp_path / "agents" / "main" / "agent"
    agent_dir.mkdir(parents=True)
    db = sqlite3.connect(agent_dir / "openclaw-agent.sqlite")
    db.executescript(
        "create table session_nodes (session_key text, current_session_id text);"
        "create table session_windows (session_id text, session_key text, previous_session_id text, reason text,"
        " created_at int, updated_at int, channel text, parent_session_key text, display_name text);"
        "create table transcript_events (session_id text, seq int, event_json text, created_at int);"
        "create table session_transcript_index_state (session_id text, leaf_event_id text);"
    )
    db.execute("insert into session_nodes values ('agent:main:explicit:s', 'win-2')")
    db.executemany(
        "insert into session_windows values (?,?,?,?,?,?,?,?,?)",
        [
            ("win-1", "agent:main:explicit:s", None, "initial", 1, 2, "cli", None, "First"),
            ("win-2", "agent:main:explicit:s", "win-1", "reset", 3, 4, "cli", None, "Second"),
        ],
    )
    db.executemany(
        "insert into transcript_events values ('win-2', ?, ?, 0)", [(seq, json.dumps(e)) for seq, e in enumerate(events)]
    )
    db.execute("insert into session_transcript_index_state values ('win-2', ?)", (leaf,))
    db.commit()
    db.close()
    return {s.source_id: s for s in read_openclaw(tmp_path)}


SIGNED = json.dumps({"type": "reasoning", "id": "rs_1", "encrypted_content": "sealed", "summary": []})


def test_the_active_branch_becomes_neutral_items(tmp_path):
    events = [
        {"type": "session", "id": "h", "version": 3, "cwd": "/w"},
        _message("u1", None, role="user", content="read the file"),
        _message(
            "a1",
            "u1",
            role="assistant",
            provider="openai",
            model="gpt-6-astra",
            api="openai-responses",
            content=[
                {"type": "thinking", "thinking": "", "thinkingSignature": SIGNED},
                {"type": "toolCall", "id": "call_1|fc_1", "name": "read", "arguments": {"path": "a"}},
            ],
        ),
        _message("t1", "a1", role="toolResult", toolCallId="call_1|fc_1", toolName="read", content=[{"type": "text", "text": "A"}]),
        _message("x1", "t1", role="user", content="abandoned branch"),
        _event("c1", "t1", type="compaction", summary="we read a", firstKeptEntryId="t1", tokensBefore=10),
        _message("a2", "c1", role="assistant", provider="openai", model="gpt-6-astra", content=[{"type": "text", "text": "ok"}]),
    ]
    sessions = _state(tmp_path, events, leaf="a2")
    second = sessions["win-2"]
    assert [item.meta["kind"] for item in second.items] == [
        "user",
        "reasoning",
        "tool_call",
        "tool_output",
        "compaction_summary",
        "text",
    ]
    reasoning, call, output = second.items[1], second.items[2], second.items[3]
    assert reasoning.item["encrypted_content"] == "sealed" and reasoning.meta["provider"] == "openai"
    assert call.item["call_id"] == output.item["call_id"] == "call_1"
    assert call.meta["tool"]["class"] == "read_file"
    assert second.turn_ends == (3, 6)
    assert second.metadata["previous_source_id"] == "win-1" and second.metadata["harness_version"] == "3"
    assert "win-1" in sessions
