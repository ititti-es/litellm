from typing import Final

from litellm.sessions.provenance import (
    Provenance,
    harness_of,
    provenance_from_stored,
    provenance_of,
    stored_fields,
)


def test_known_clients_are_named_and_others_keep_their_product_token() -> None:
    assert harness_of("openclaw/2026.9.6 (linux)") == "openclaw"
    assert harness_of("opencode/latest/2.0.12/cli") == "opencode"
    assert harness_of("claude-cli/2.1.0 (external, cli)") == "claude_code"
    assert harness_of("codex_cli_rs/0.40.0") == "codex"
    assert harness_of("LangGraph/0.4 (python)") == "langgraph"
    assert harness_of("Python-urllib/3.14") is None
    assert harness_of("OpenAI/JS 7.15.0") is None
    assert harness_of("") is None
    assert harness_of(None) is None


def test_an_explicit_header_beats_the_key_and_the_user_agent() -> None:
    metadata: Final = {
        "headers": {"X-OpenClaw-Agent-Id": "Sett", "x-litellm-harness": "openclaw"},
        "user_api_key_metadata": {"agent": "other", "harness": "keyed"},
        "user_agent": "opencode/latest/2.0.12/cli",
    }
    assert provenance_of(metadata) == Provenance(agent="sett", harness="openclaw")


def test_the_key_names_the_agent_and_the_user_agent_names_the_harness_when_headers_are_silent() -> None:
    metadata: Final = {"headers": {}, "user_api_key_metadata": {"agent": "sett"}, "user_agent": "openclaw/2026.9.6"}
    assert provenance_of(metadata) == Provenance(agent="sett", harness="openclaw")


def test_a_request_that_says_nothing_has_no_provenance() -> None:
    assert provenance_of({"headers": {}, "user_agent": ""}) is None
    assert provenance_of(None) is None


def test_header_values_are_reduced_to_a_short_label() -> None:
    metadata: Final = {"headers": {"x-litellm-agent": "  Sett; drop table x" + "y" * 100}}
    found: Final = provenance_of(metadata)
    assert found is not None and found.agent == "sett"


def test_a_later_turn_fills_only_what_it_says() -> None:
    earlier: Final = Provenance(agent="sett", harness="openclaw")
    assert Provenance(harness="opencode").over(earlier) == Provenance(agent="sett", harness="opencode")
    assert Provenance().over(earlier) == earlier
    assert Provenance().over(None) is None


def test_stored_fields_round_trip() -> None:
    provenance: Final = Provenance(agent="sett", harness="openclaw")
    assert provenance_from_stored(stored_fields(provenance)) == provenance
    assert stored_fields(Provenance(harness="opencode")) == {"harness": "opencode"}
    assert provenance_from_stored({"litellm_session_id": "ses_1"}) is None
