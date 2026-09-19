from __future__ import annotations

from typing import Final

from litellm.proxy._types import SessionManagerSettings
from litellm.sessions import SessionManager


def configured_session_manager(raw: object, current: SessionManager | None) -> SessionManager | None:
    """The manager general_settings.session_manager asks for, keeping the live one (and its sessions) across reloads"""
    if raw is None:
        return None
    settings: Final = SessionManagerSettings.model_validate(raw)
    if not settings.enabled:
        return None
    budget: Final = settings.budget_mb << 20
    if current is not None and current.budget_bytes == budget:
        return current
    return SessionManager(budget_bytes=budget)
