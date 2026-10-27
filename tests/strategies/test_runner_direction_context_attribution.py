"""Regression coverage for fresh-but-directionless LIVE market context.

These checks never enable order placement or substitute an option-derived bias.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _context(**values: object) -> dict[str, object]:
    return {"timestamp": 995.0, **values}


def _runner_with_context(monkeypatch, *, spot: dict, futures: dict) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._strategy_manager = SimpleNamespace(
        _latest_context_snapshots={
            "spot_context": spot,
            "futures_context": futures,
        },
        _live_context_max_age_seconds=lambda: 60.0,
    )
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.time", lambda: 1000.0)
    return runner


def test_fresh_underlying_quotes_without_direction_do_not_authorize_a_side(
    monkeypatch,
) -> None:
    runner = _runner_with_context(
        monkeypatch,
        spot=_context(role="spot_context", direction_context_reasons=["neutral"]),
        futures=_context(
            role="futures_context", direction_context_reasons=["range_unresolved"]
        ),
    )

    result = runner._underlying_context_from_strategy_manager()

    assert result["spot_fresh"] is True
    assert result["futures_fresh"] is True
    assert result["context_fresh"] is True
    assert result["direction_bias"] is None
    assert result["underlying_direction_bias"] is None
    assert result["underlying_direction_confidence"] is None
    assert "direction_context_conflict" not in result


def test_fresh_opposing_spot_futures_direction_does_not_choose_spot(
    monkeypatch,
) -> None:
    runner = _runner_with_context(
        monkeypatch,
        spot=_context(role="spot_context", underlying_direction_bias="CE"),
        futures=_context(role="futures_context", underlying_direction_bias="PE"),
    )

    result = runner._underlying_context_from_strategy_manager()

    assert result["direction_context_conflict"] is True
    assert result["direction_bias"] is None
    assert result["underlying_direction_bias"] is None


def test_one_valid_underlying_source_can_still_supply_direction(
    monkeypatch,
) -> None:
    runner = _runner_with_context(
        monkeypatch,
        spot=_context(role="spot_context", underlying_direction_bias="CE"),
        futures=_context(role="futures_context"),
    )

    result = runner._underlying_context_from_strategy_manager()

    assert result["direction_bias"] == "CE"
    assert result["underlying_direction_bias"] == "CE"
    assert result["direction_context_source"] == "spot_context"


def test_direction_skip_log_carries_underlying_provenance_and_still_rejects() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    start = source.index('f"direction_context_blocked:{symbol}:{direction_reason}"')
    end = source.index("self._emit_runner_eval_decision(", start)
    diagnostic = source[start:end]
    for marker in (
        "spot_bias=%s",
        "fut_bias=%s",
        "spot_direction_reasons=%s",
        "fut_direction_reasons=%s",
        '"spot_bias": spot_bias',
        '"fut_bias": futures_bias',
        '"spot_direction_reasons": spot_direction_reasons',
        '"fut_direction_reasons": futures_direction_reasons',
        "level=logging.WARNING",
    ):
        assert marker in diagnostic
    decision = source[end : end + 550]
    assert "allowed=False" in decision
    assert "return" in decision
