from __future__ import annotations

import pytest

from nifty_scalper_bot.backtesting.completed_trade_analysis import (
    attribution_readiness,
    canonicalize_completed_trades,
    chronological_post_cost_blocks,
    summarize_completed_trades,
)


def _trade(
    trade_id: str,
    closed_at: float,
    *,
    strategy: str = "VWAPPro",
    gross_pnl: float = 100.0,
    estimated_costs: float = 20.0,
    net_pnl: float = 80.0,
    ledger_complete: bool = True,
    state: str = "CLOSED",
    signal_quality: dict[str, float] | None = None,
) -> dict[str, object]:
    outcome: dict[str, object] = {}
    if signal_quality is not None:
        outcome["signal_quality"] = signal_quality
    return {
        "trade_id": trade_id,
        "closed_at": closed_at,
        "strategy": strategy,
        "gross_pnl": gross_pnl,
        "estimated_costs": estimated_costs,
        "net_pnl": net_pnl,
        "ledger_complete": ledger_complete,
        "state": state,
        "outcome": outcome,
    }


def test_canonicalize_filters_and_orders_complete_closed_rows() -> None:
    rows = [
        _trade("later", 30.0),
        _trade("open", 20.0, state="OPEN"),
        _trade("incomplete", 10.0, ledger_complete=False),
        _trade("earlier", 5.0),
    ]

    result = canonicalize_completed_trades(rows)

    assert [trade.trade_id for trade in result] == ["earlier", "later"]


def test_canonicalize_completed_trades_rejects_duplicate_trade_identity() -> None:
    rows = [_trade("duplicate", 1.0), _trade("duplicate", 2.0)]

    with pytest.raises(ValueError, match="duplicate trade_id"):
        canonicalize_completed_trades(rows)


def test_canonicalize_completed_trades_rejects_invalid_post_cost_identity() -> None:
    rows = [
        _trade(
            "bad-costs",
            1.0,
            gross_pnl=100.0,
            estimated_costs=20.0,
            net_pnl=60.0,
        )
    ]

    with pytest.raises(ValueError, match="gross_pnl - estimated_costs"):
        canonicalize_completed_trades(rows)


def test_chronological_post_cost_blocks_are_contiguous_and_non_overlapping() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                f"t{index}",
                float(index),
                net_pnl=float(index),
                gross_pnl=float(index + 20),
            )
            for index in range(1, 7)
        ]
    )

    blocks = chronological_post_cost_blocks(trades, block_size=2)

    assert [(block.start_trade_id, block.end_trade_id) for block in blocks] == [
        ("t1", "t2"),
        ("t3", "t4"),
        ("t5", "t6"),
    ]
    assert [block.summary.trade_count for block in blocks] == [2, 2, 2]


def test_summary_uses_post_cost_net_pnl() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade(
                "winner",
                1.0,
                gross_pnl=120.0,
                estimated_costs=20.0,
                net_pnl=100.0,
            ),
            _trade(
                "loser",
                2.0,
                gross_pnl=-30.0,
                estimated_costs=20.0,
                net_pnl=-50.0,
            ),
        ]
    )

    summary = summarize_completed_trades(trades)

    assert summary.gross_pnl == 90.0
    assert summary.estimated_costs == 40.0
    assert summary.net_pnl == 50.0
    assert summary.expectancy == 25.0
    assert summary.win_rate == 0.5
    assert summary.profit_factor == 2.0


def test_attribution_readiness_fails_closed_for_missing_component_or_quality() -> None:
    trades = canonicalize_completed_trades(
        [
            _trade("vwap", 1.0, strategy="VWAPPro"),
            _trade(
                "smc",
                2.0,
                strategy="SMC",
                signal_quality={"alpha_score": 8.0, "strategy_score": 7.5},
            ),
        ]
    )

    readiness = attribution_readiness(
        trades,
        required_components=("ORBPro", "SMC", "VWAPPro"),
    )

    assert readiness.ready is False
    assert readiness.coverage["ORBPro"].completed_trades == 0
    assert readiness.coverage["VWAPPro"].with_signal_quality == 0
    assert "missing_completed_trades:ORBPro" in readiness.blockers
    assert "missing_signal_quality:VWAPPro" in readiness.blockers


def test_attribution_readiness_accepts_complete_component_evidence() -> None:
    quality = {"alpha_score": 8.0, "strategy_score": 7.5}
    trades = canonicalize_completed_trades(
        [
            _trade("orb", 1.0, strategy="ORBPro", signal_quality=quality),
            _trade("smc", 2.0, strategy="SMC", signal_quality=quality),
            _trade("vwap", 3.0, strategy="VWAPPro", signal_quality=quality),
        ]
    )

    readiness = attribution_readiness(
        trades,
        required_components=("ORBPro", "SMC", "VWAPPro"),
    )

    assert readiness.ready is True
    assert readiness.blockers == ()
