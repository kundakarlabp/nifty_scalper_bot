from __future__ import annotations

from pathlib import Path


def test_single_candidate_screen_uses_canonical_tick_age_resolver() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    marker = "EXECUTION_SINGLE_CANDIDATE_SCREEN"
    block_start = source.index(marker)
    block_end = source.index("ranked_candidates =", block_start)
    block = source[block_start:block_end]

    assert "resolve_tick_age_ms(" in block
    assert "lone.get(\"tick_age_s\")" not in block
