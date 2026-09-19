from litellm.sessions.cache_clock import (
    CacheClock,
    CacheSample,
    classify,
    estimate_ttl,
    prior_ttl,
    retention_hint,
)


def test_a_turn_counts_as_a_hit_or_miss_only_when_its_usage_can_tell():
    assert classify(prefix_tokens=10_000, cached_tokens=9_000) is True
    assert classify(prefix_tokens=10_000, cached_tokens=0) is False
    assert classify(prefix_tokens=10_000, cached_tokens=2_000) is None
    assert classify(prefix_tokens=500, cached_tokens=0) is None


def test_without_evidence_the_estimate_stays_inside_the_documented_ttl():
    assert estimate_ttl((), prior_seconds=300.0, target=0.8) == 270.0
    assert CacheClock().estimate("xai", "grok").ttl_seconds == 270.0


def test_repeated_hits_after_long_gaps_extend_the_estimate():
    clock = CacheClock()
    for _ in range(40):
        clock.observe("openai", "gpt", gap_seconds=900.0, prefix_tokens=20_000, cached_tokens=19_000)
    estimate = clock.estimate("openai", "gpt")
    assert estimate.ttl_seconds == 900.0
    assert (estimate.samples, estimate.hits) == (40, 40)


def test_a_few_long_hits_do_not_outweigh_the_prior():
    clock = CacheClock()
    clock.observe("openai", "gpt", gap_seconds=900.0, prefix_tokens=20_000, cached_tokens=19_000)
    assert clock.estimate("openai", "gpt").ttl_seconds == 270.0


def test_misses_inside_the_documented_window_shorten_the_estimate():
    clock = CacheClock()
    for gap in (30.0, 60.0, 90.0):
        for _ in range(10):
            clock.observe("xai", "grok", gap_seconds=gap, prefix_tokens=20_000, cached_tokens=19_000)
    for _ in range(20):
        clock.observe("xai", "grok", gap_seconds=120.0, prefix_tokens=20_000, cached_tokens=0)
    assert clock.estimate("xai", "grok").ttl_seconds == 90.0


def test_models_are_learned_separately():
    clock = CacheClock()
    for _ in range(20):
        clock.observe("xai", "a", gap_seconds=60.0, prefix_tokens=20_000, cached_tokens=0)
    assert clock.estimate("xai", "a").ttl_seconds < 270.0
    assert clock.estimate("xai", "b").ttl_seconds == 270.0


def test_the_fit_never_credits_a_gap_past_the_first_confident_miss():
    samples = (CacheSample(100.0, True), CacheSample(200.0, False, 10.0), CacheSample(250.0, True))
    assert estimate_ttl(samples, prior_seconds=300.0, target=0.8) == 100.0


def test_explicit_retention_raises_the_prior():
    one_hour = retention_hint({}, ({"content": [{"type": "input_text", "cache_control": {"ttl": "1h"}}]},))
    assert one_hour == 3600.0
    assert retention_hint({"prompt_cache_retention": "24h"}, ()) == 86_400.0
    assert retention_hint({}, ({"type": "message"},)) is None
    assert prior_ttl("anthropic", one_hour) == 3600.0
    assert prior_ttl("anthropic") == 300.0


def test_old_evidence_fades_so_the_estimate_follows_current_load():
    now = [0.0]
    clock = CacheClock(half_life_seconds=600.0, now=lambda: now[0])
    for _ in range(40):
        clock.observe("xai", "grok", gap_seconds=900.0, prefix_tokens=20_000, cached_tokens=19_000)
    assert clock.estimate("xai", "grok").ttl_seconds == 900.0
    now[0] = 6 * 3600.0
    assert clock.estimate("xai", "grok").ttl_seconds == 270.0


def test_recent_misses_outweigh_older_hits_at_the_same_gap():
    now = [0.0]
    clock = CacheClock(half_life_seconds=600.0, now=lambda: now[0])
    for _ in range(40):
        clock.observe("xai", "grok", gap_seconds=240.0, prefix_tokens=20_000, cached_tokens=19_000)
    now[0] = 3600.0
    for _ in range(10):
        clock.observe("xai", "grok", gap_seconds=200.0, prefix_tokens=20_000, cached_tokens=0)
    assert clock.estimate("xai", "grok").ttl_seconds < 200.0
