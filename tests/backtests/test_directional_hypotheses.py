"""Deterministic evidence and safety checks for non-live strategy hypotheses."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from nifty_scalper_bot.backtesting.directional_hypotheses import (
    ArchivedDirectionalHypothesis,
    classify_completed_futures_bars,
    run_directional_hypothesis_research,
)

_START = datetime(2026, 10, 1, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
_FUTURE = "NFO:NIFTY26OCTFUT"
_CE = "NFO:NIFTY26O0624000CE"
_PE = "NFO:NIFTY26O0624000PE"


def _bar(index: int, close: float, *, volume: float = 1000.0) -> dict[str, Any]:
    return {
        "timestamp": _START + timedelta(minutes=index),
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
        "is_complete": True,
        "is_provisional": False,
    }


def _momentum_bars(side: str = "CE") -> list[dict[str, Any]]:
    sign = 1 if side == "CE" else -1
    rows = [_bar(i, 24000 + sign * i * 2) for i in range(45)]
    rows[-1] = _bar(44, 24000 + sign * (43 * 2 + 12), volume=2000)
    return rows


def test_momentum_acceleration_respects_futures_side_and_completed_bar() -> None:
    for side in ("CE", "PE"):
        rows = _momentum_bars(side)
        assert classify_completed_futures_bars(rows)["momentum_acceleration"] == side

        rows[-1] = {**rows[-1], "is_complete": False}
        assert classify_completed_futures_bars(rows) == {}


def test_compression_is_based_on_prior_bands_not_breakout_candle() -> None:
    rows = [_bar(i, 24000.0) for i in range(45)]
    rows[-1] = _bar(44, 24005.0, volume=2000.0)
    assert classify_completed_futures_bars(rows)["compression_breakout"] == "CE"
    rows[-1] = _bar(44, 24000.0, volume=2000.0)
    assert "compression_breakout" not in classify_completed_futures_bars(rows)


def test_trend_pullback_requires_reclaim_after_retracement() -> None:
    rows = [_bar(i, 24000.0 + 2.0 * i) for i in range(45)]
    rows[-2] = {
        **_bar(43, 24076.0),
        "open": 24083.0,
        "high": 24084.0,
        "low": 24075.0,
    }
    rows[-1] = {
        **_bar(44, 24095.0, volume=1500.0),
        "open": 24077.0,
        "low": 24076.0,
    }
    assert classify_completed_futures_bars(rows)["trend_pullback"] == "CE"
    assert "trend_pullback" not in classify_completed_futures_bars(rows[:-1])


def test_disconnected_or_ambiguous_history_fails_closed() -> None:
    rows = _momentum_bars()
    assert classify_completed_futures_bars(rows[:-1]) == {}
    bad = list(rows)
    bad[20] = {**bad[20], "timestamp": bad[20]["timestamp"] + timedelta(minutes=2)}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "timestamp": bad[-1]["timestamp"].replace(tzinfo=None)}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "close": float("nan")}
    assert classify_completed_futures_bars(bad) == {}
    bad = list(rows)
    bad[-1] = {**bad[-1], "is_provisional": True}
    assert classify_completed_futures_bars(bad) == {}


class _HistoryEngine:
    def __init__(
        self,
        bars: list[dict[str, Any]],
        future_symbol: str = _FUTURE,
    ) -> None:
        self.bars = bars
        self.future_symbol = future_symbol

    def get_history(
        self, symbol: str, count: int, *, field: str
    ) -> list[dict[str, Any]]:
        assert field == "bars"
        return self.bars[-count:] if symbol == self.future_symbol else []

    def get_atr(self, symbol: str) -> float | None:
        return 2.0 if symbol in {_CE, _PE} else None


def test_archived_adapter_only_generates_option_buy_with_next_bar_proxy() -> None:
    rows = _momentum_bars()
    candidate = ArchivedDirectionalHypothesis(
        "momentum_acceleration", _HistoryEngine(rows)
    )
    context = {"futures_symbol": _FUTURE, "bar_timestamp": rows[-1]["timestamp"]}

    assert candidate.generate_signal(_PE, context, 60.0) is None
    assert candidate.generate_signal("NSE:NIFTY 50", context, 60.0) is None
    assert (
        candidate.generate_signal(
            _CE, {**context, "bar_timestamp": rows[-2]["timestamp"]}, 60.0
        )
        is None
    )

    signal = candidate.generate_signal(_CE, context, 60.0)
    assert signal is not None
    assert signal.action == "BUY"
    assert signal.stop_loss == pytest.approx(57.6)
    assert signal.take_profit == pytest.approx(64.8)
    assert signal.metadata["signal_family"] == "research_only"
    assert signal.metadata["contract_side"] == "CE"
    assert signal.metadata["setup_id"].startswith("research:momentum_acceleration:")


def _archive(directory: Path) -> None:
    candles = directory / "candles"
    candles.mkdir()
    for symbol, kind in (
        ("NSE:NIFTY 50", "EQ"),
        (_FUTURE, "FUT"),
        (_CE, "CE"),
        (_PE, "PE"),
    ):
        payload = {
            "symbol": symbol,
            "timestamp_convention": "bar_start",
            "instrument": {
                "instrument_type": kind,
                "name": "NIFTY",
                "lot_size": 75,
                "expiry": "2026-10-06",
            },
            "candles": [
                [
                    (_START + timedelta(minutes=i)).isoformat(),
                    100.0,
                    102.0,
                    98.0,
                    100.0,
                    1000.0,
                ]
                for i in range(4)
            ],
        }
        (candles / f"{kind}.json").write_text(json.dumps(payload))


def test_hypothesis_research_reuses_canonical_costed_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    _archive(tmp_path)

    result = run_directional_hypothesis_research(tmp_path)
    assert result["live_equivalent"] is False
    assert result["selected_for_live"] is None
    assert result["promotion_eligible"] is False
    assert result["evidence_label"] == "RESEARCH_CANDIDATE"
    assert len(result["scenarios"]) == 3
    assert [row["slippage_bps_per_side"] for row in result["scenarios"]] == [
        10.0,
        25.0,
        50.0,
    ]
    for row in result["scenarios"]:
        assert set(row["strategies"]) == {
            "PullbackResearch",
            "CompressionResearch",
            "MomentumResearch",
        }
        assert all(
            value["metrics"]["trade_count"] == 0 for value in row["strategies"].values()
        )

    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(ValueError, match="research_requires_shadow_process"):
        run_directional_hypothesis_research(tmp_path)


def test_exact_public_archive_future_context_is_research_only() -> None:
    """Monthly public archives use a continuous context future identifier."""
    rows = _momentum_bars()
    historical_symbol = "NFO:NIFTY_F1_CONTEXT"
    candidate = ArchivedDirectionalHypothesis(
        "momentum_acceleration",
        _HistoryEngine(rows, future_symbol=historical_symbol),
    )
    context = {
        "futures_symbol": historical_symbol,
        "bar_timestamp": rows[-1]["timestamp"],
    }
    assert candidate.generate_signal(_CE, context, 60.0) is not None
    assert (
        candidate.generate_signal(
            _CE, {**context, "futures_symbol": "NFO:NIFTY_OTHER_CONTEXT"}, 60.0
        )
        is None
    )
