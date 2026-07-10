from __future__ import annotations

from pathlib import Path


def test_execution_candidate_ranking_uses_canonical_tick_age_resolver() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    start = source.index('score_raw = metadata.get("candidate_score")')
    end = source.index("depth_available = bool(", start)
    block = source[start:end]

    assert "resolve_tick_age_ms(" in block
    assert "999000.0" not in block
