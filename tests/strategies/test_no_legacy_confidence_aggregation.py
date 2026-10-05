from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.strategies import runner


def test_runner_has_no_legacy_confidence_weighted_signal_aggregation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "def aggregate_signals_by_symbol(" not in source
    assert "def _normalize_confidence(" not in source
    assert "def _passes_spot_trend_filter(" not in source
    assert "Confidence threshold" not in source
    assert "VWAP filter" not in source
