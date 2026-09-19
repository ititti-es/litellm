from litellm.sessions.compaction_thresholds import ThresholdOverrides, Thresholds, layered, worth_compacting

WINDOW = 256_000


def test_a_rich_but_modest_session_is_never_compacted():
    bars = Thresholds(min_prefix_tokens=20_000, min_context_fraction=0.5, context_window=WINDOW)
    assert not worth_compacting(20_000, bars)
    assert not worth_compacting(127_999, bars)
    assert worth_compacting(128_000, bars)


def test_an_unknown_window_never_clears_a_fraction_bar():
    assert not worth_compacting(900_000, Thresholds(min_context_fraction=0.5))
    assert worth_compacting(900_000, Thresholds(min_prefix_tokens=100_000))


def test_layers_override_field_by_field_and_can_switch_compaction_off():
    base = Thresholds(min_prefix_tokens=20_000, min_context_fraction=0.5, context_window=WINDOW)
    model = ThresholdOverrides(min_context_fraction=0.8, context_window=128_000)
    policy = ThresholdOverrides(min_prefix_tokens=50_000)
    merged = layered(base, model, None, policy)
    assert merged == Thresholds(min_prefix_tokens=50_000, min_context_fraction=0.8, context_window=128_000)
    assert not worth_compacting(100_000, merged) and worth_compacting(102_400, merged)
    assert not worth_compacting(10**9, layered(base, ThresholdOverrides(enabled=False)))
