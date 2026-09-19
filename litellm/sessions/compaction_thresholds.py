"""When a session is big enough to be worth compacting

A compaction trades verbatim history for a summary. For a modest session that trade is a loss: resending it cold
after a break is cheap, and the detail is worth more than the tokens saved. So a session is compacted only when
its next prefix clears every configured bar:

- `min_prefix_tokens`: an absolute floor
- `min_context_fraction`: a share of the model's context window (0.5 = half full); a session on a model whose
  window is unknown does not clear it, so an unknown model is never compacted by this rule

Bars come from, most specific first: the session's compaction policy, the model's entry in the proxy settings,
the proxy-wide settings. Each field falls back independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Thresholds:
    enabled: bool = True
    min_prefix_tokens: int = 0
    min_context_fraction: float | None = None
    context_window: int | None = None


@dataclass(frozen=True, slots=True)
class ThresholdOverrides:
    """One layer of configuration; None leaves the field to the layer below"""

    enabled: bool | None = None
    min_prefix_tokens: int | None = None
    min_context_fraction: float | None = None
    context_window: int | None = None


def layered(base: Thresholds, *layers: ThresholdOverrides | None) -> Thresholds:
    """*base* with each layer applied in turn, later layers winning field by field"""
    result: Thresholds = base  # rebind-ok: folds the layers in order
    for layer in layers:
        if layer is None:
            continue
        result = Thresholds(
            enabled=layer.enabled if layer.enabled is not None else result.enabled,
            min_prefix_tokens=(
                layer.min_prefix_tokens if layer.min_prefix_tokens is not None else result.min_prefix_tokens
            ),
            min_context_fraction=(
                layer.min_context_fraction if layer.min_context_fraction is not None else result.min_context_fraction
            ),
            context_window=layer.context_window if layer.context_window is not None else result.context_window,
        )
    return result


def worth_compacting(prefix_tokens: int, thresholds: Thresholds) -> bool:
    if not thresholds.enabled or prefix_tokens < thresholds.min_prefix_tokens:
        return False
    fraction: Final = thresholds.min_context_fraction
    if fraction is None:
        return True
    window: Final = thresholds.context_window
    return window is not None and window > 0 and prefix_tokens >= fraction * window
