"""How long an upstream prompt cache stays warm, learned per provider and model

Every continued turn is an experiment: the gap since the previous request (which wrote or refreshed the prefix)
and whether the provider reported that prefix as cached. A hit at gap g says the cache outlived g; a miss says it
did not. The estimate is the longest gap at which a monotone fit of the hit rate still clears the target, seeded
with pseudo-observations from each provider's documented TTL so a new model starts from the published figure
and only moves when real evidence outweighs it.

Some providers (xAI among them) keep prefixes longer or shorter depending on capacity against load, so the
lifetime drifts. Evidence therefore decays with age: the estimate follows current conditions, and returns to
the prior when there is no recent traffic to learn from.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# Documented or observed minimum warm windows, in seconds; the default covers providers that publish none
_PRIOR_TTL: Final = MappingProxyType(
    {
        "anthropic": 300.0,
        "bedrock": 300.0,
        "vertex_ai": 300.0,
        "openai": 300.0,
        "azure": 300.0,
        "xai": 300.0,
        "gemini": 300.0,
    }
)
_DEFAULT_PRIOR_TTL: Final = 300.0
_HIT_FRACTION: Final = 0.5
_MISS_FRACTION: Final = 0.05
_PRIOR_WEIGHT: Final = 4.0
_PRIOR_HIT_AT: Final = 0.9
_PRIOR_MISS_AT: Final = 2.0
_WINDOW: Final = 256


@dataclass(frozen=True, slots=True)
class CacheSample:
    gap_seconds: float
    hit: bool
    weight: float = 1.0
    observed_at: float = 0.0


@dataclass(frozen=True, slots=True)
class CacheEstimate:
    ttl_seconds: float
    prior_seconds: float
    samples: int
    hits: int


def classify(prefix_tokens: int, cached_tokens: int, min_cacheable: int) -> bool | None:
    """Whether a turn found the previous request's prefix cached; None when the usage cannot tell

    A prefix below what the model will cache at all (litellm.utils.get_prompt_cache_min_tokens) says
    nothing about how long the cache lives
    """
    if prefix_tokens < min_cacheable:
        return None
    if cached_tokens >= prefix_tokens * _HIT_FRACTION:
        return True
    if cached_tokens <= prefix_tokens * _MISS_FRACTION:
        return False
    return None


def prior_ttl(provider: str, retention_hint: float | None = None, priors: Mapping[str, float] = _PRIOR_TTL) -> float:
    """A request that asked for longer retention (Anthropic's 1h breakpoints, OpenAI's extended retention) says so"""
    base: Final = priors.get(provider, _PRIOR_TTL.get(provider, _DEFAULT_PRIOR_TTL))
    return max(base, retention_hint) if retention_hint is not None else base


def retention_hint(params: Mapping[str, object]) -> float | None:
    """Longer retention a request asked for explicitly

    Only what the Responses request itself carries. Anthropic's `cache_control` breakpoints and their
    5m/1h TTLs are parsed by litellm.llms.anthropic.prompt_cache_prediction, which owns that reading.
    """
    return 86_400.0 if params.get("prompt_cache_retention") == "24h" else None


def estimate_ttl(samples: Sequence[CacheSample], prior_seconds: float, target: float) -> float:
    """The longest gap whose fitted hit rate is at least *target*; never beyond the longest observed hit"""
    seeded: Final = sorted(
        (
            *samples,
            CacheSample(prior_seconds * _PRIOR_HIT_AT, True, _PRIOR_WEIGHT),
            CacheSample(prior_seconds * _PRIOR_MISS_AT, False, _PRIOR_WEIGHT),
        ),
        key=lambda sample: sample.gap_seconds,
    )
    fitted: Final = _nonincreasing_fit(seeded)
    safe: Final = tuple(sample.gap_seconds for sample, rate in zip(seeded, fitted) if rate >= target and sample.hit)
    return max(safe) if safe else 0.0


def _nonincreasing_fit(samples: Sequence[CacheSample]) -> tuple[float, ...]:
    """Pool-adjacent-violators: the weighted least-squares hit rate that never rises with the gap"""
    blocks: Final[list[tuple[float, float, int]]] = []  # mutable-ok: PAV stack of (rate, weight, count)
    for sample in samples:
        rate: float = 1.0 if sample.hit else 0.0  # rebind-ok: pooled into the block below
        weight: float = sample.weight  # rebind-ok: pooled into the block below
        count: int = 1  # rebind-ok: pooled into the block below
        while blocks and blocks[-1][0] < rate:
            prior_rate, prior_weight, prior_count = blocks.pop()
            rate = (prior_rate * prior_weight + rate * weight) / (prior_weight + weight)
            weight += prior_weight
            count += prior_count
        blocks.append((rate, weight, count))
    return tuple(rate for rate, _, count in blocks for _ in range(count))


def min_cacheable_tokens(model: str) -> int:
    """The smallest prefix this model caches at all, from LiteLLM's per-model cost map"""
    from litellm.utils import get_prompt_cache_min_tokens

    return get_prompt_cache_min_tokens(model=model)


class CacheClock:
    """Recent cache observations per (provider, model), and the warm window they imply"""

    def __init__(
        self,
        target: float = 0.8,
        priors: Mapping[str, float] | None = None,
        half_life_seconds: float = 3600.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._target: Final = target
        self._half_life: Final = half_life_seconds
        self._now: Final = now
        self._priors: Final = MappingProxyType(dict(priors)) if priors is not None else _PRIOR_TTL
        self._samples: Final[dict[tuple[str, str], deque[CacheSample]]] = {}  # mutable-ok: bounded window per model

    def observe(
        self,
        provider: str,
        model: str,
        gap_seconds: float,
        prefix_tokens: int,
        cached_tokens: int,
        observed_at: float | None = None,
    ) -> None:
        hit: Final = classify(prefix_tokens, cached_tokens, min_cacheable_tokens(model))
        if hit is None or gap_seconds < 0:
            return
        window: Final = self._samples.setdefault((provider, model), deque(maxlen=_WINDOW))  # mutable-ok: bounded window
        window.append(CacheSample(gap_seconds, hit, observed_at=self._now() if observed_at is None else observed_at))

    def estimate(self, provider: str, model: str, retention_hint: float | None = None) -> CacheEstimate:
        samples: Final = tuple(self._samples.get((provider, model), ()))
        prior: Final = prior_ttl(provider, retention_hint, self._priors)
        now: Final = self._now()
        weighted: Final = tuple(
            CacheSample(sample.gap_seconds, sample.hit, self._decayed(now - sample.observed_at)) for sample in samples
        )
        return CacheEstimate(
            ttl_seconds=estimate_ttl(weighted, prior, self._target),
            prior_seconds=prior,
            samples=len(samples),
            hits=sum(1 for sample in samples if sample.hit),
        )

    def _decayed(self, age_seconds: float) -> float:
        return 0.5 ** (max(age_seconds, 0.0) / self._half_life)
