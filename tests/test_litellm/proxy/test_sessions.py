import pydantic
import pytest

from litellm.proxy.sessions import configured_session_manager


def test_a_reload_with_the_same_settings_keeps_the_live_sessions():
    first = configured_session_manager({"enabled": True, "budget_mb": 8}, None)
    assert first is not None and first.budget_bytes == 8 << 20
    assert configured_session_manager({"enabled": True, "budget_mb": 8}, first) is first
    resized = configured_session_manager({"enabled": True, "budget_mb": 16}, first)
    assert resized is not first and resized is not None and resized.budget_bytes == 16 << 20


def test_absent_or_disabled_settings_turn_the_manager_off():
    live = configured_session_manager({"enabled": True}, None)
    assert configured_session_manager(None, live) is None
    assert configured_session_manager({"enabled": False}, live) is None


def test_a_non_positive_budget_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        configured_session_manager({"enabled": True, "budget_mb": 0}, None)
