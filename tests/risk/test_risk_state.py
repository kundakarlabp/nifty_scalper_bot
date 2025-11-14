from __future__ import annotations

from nifty_scalper_bot.risk.risk_manager import RiskState


class _StubDataHub:
    def __init__(self, fresh: bool = True) -> None:
        self.fresh = fresh
        self.calls: list[str] = []

    def normalize(self, symbol: str) -> str:
        return str(symbol or "").strip().upper()

    def is_fresh(self, symbol: str) -> tuple[bool, dict[str, object]]:
        self.calls.append(symbol)
        return self.fresh, {
            "threshold_ms": 1000.0,
            "reason": None if self.fresh else "stale",
            "effective_ms": 500.0,
        }


def test_stale_quote_blocks_trading() -> None:
    state = RiskState(
        quote_stale_ms_threshold=100,
        spread_widen_mult=3.0,
        max_intraday_dd=-1_000.0,
        max_consecutive_losses=3,
    )
    state.on_tick(100.0, 101.0, ts_ns=0)
    allowed, reasons = state.should_trade(now_ns=50_000_000)
    assert allowed is True
    allowed, reasons = state.should_trade(now_ns=200_000_000)
    assert allowed is False
    assert "STALE" in reasons


def test_spread_widening_triggers_block() -> None:
    state = RiskState(
        quote_stale_ms_threshold=500,
        spread_widen_mult=3.0,
        max_intraday_dd=-5_000.0,
        max_consecutive_losses=3,
    )
    state.on_tick(100.0, 101.0, ts_ns=0)
    state.on_tick(101.0, 102.0, ts_ns=200_000_000)
    state.on_tick(102.0, 103.0, ts_ns=400_000_000)
    state.on_tick(95.0, 110.0, ts_ns=600_000_000)
    allowed, reasons = state.should_trade(now_ns=600_000_000)
    assert allowed is False
    assert "WIDE_SPREAD" in reasons


def test_drawdown_and_loss_streak_blocks() -> None:
    state = RiskState(
        quote_stale_ms_threshold=500,
        spread_widen_mult=3.0,
        max_intraday_dd=-1_000.0,
        max_consecutive_losses=2,
    )
    state.on_tick(100.0, 101.0, ts_ns=0)
    state.on_trade_update(realized_pnl=0.0, unrealized_pnl=0.0)
    state.on_trade_update(realized_pnl=-600.0, unrealized_pnl=0.0)
    state.on_trade_update(realized_pnl=-1_200.0, unrealized_pnl=0.0)
    allowed, reasons = state.should_trade(now_ns=1_000_000)
    assert allowed is False
    assert "DD" in reasons
    assert "LOSER_STREAK" in reasons
    state.on_trade_update(realized_pnl=500.0, unrealized_pnl=0.0)
    allowed, reasons = state.should_trade(now_ns=2_000_000)
    assert allowed is True


def test_risk_state_uses_data_hub_freshness() -> None:
    state = RiskState(
        quote_stale_ms_threshold=50,
        spread_widen_mult=3.0,
        max_intraday_dd=-500.0,
        max_consecutive_losses=3,
    )
    hub = _StubDataHub(fresh=False)
    state.attach_data_hub(hub, symbol="nifty")

    allowed, reasons = state.should_trade()

    assert allowed is False
    assert "STALE" in reasons

    hub.fresh = True
    allowed_after, reasons_after = state.should_trade()

    assert allowed_after is True
    assert "STALE" not in reasons_after
    assert hub.calls
