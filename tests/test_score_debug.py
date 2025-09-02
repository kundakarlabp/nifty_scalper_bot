import pandas as pd
import numpy as np
import pytest

from src.config import StrategySettings
from src.strategies.scalping_strategy import EnhancedScalpingStrategy
from src.strategies.runner import StrategyRunner


class DummyTelegram:
    def send_message(self, msg: str) -> None:  # pragma: no cover - no behavior
        pass


@pytest.fixture
def strategy_config() -> StrategySettings:
    return StrategySettings(
        min_signal_score=0.0,
        confidence_threshold=0.0,
        atr_period=14,
        atr_sl_multiplier=1.5,
        atr_tp_multiplier=3.0,
    )


def _create_df(length: int = 100) -> pd.DataFrame:
    prices = np.linspace(100.0, 120.0, length)
    data = {
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=length),
    }
    index = pd.date_range(start="2023-01-01", periods=length, freq="min")
    return pd.DataFrame(data, index=index)


def test_score_breakdown_present(monkeypatch: pytest.MonkeyPatch, strategy_config: StrategySettings) -> None:
    strategy = EnhancedScalpingStrategy(
        min_signal_score=strategy_config.min_signal_score,
        confidence_threshold=strategy_config.confidence_threshold,
        atr_period=strategy_config.atr_period,
        atr_sl_multiplier=strategy_config.atr_sl_multiplier,
        atr_tp_multiplier=strategy_config.atr_tp_multiplier,
    )
    df = _create_df()
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.fetch_quote_with_depth",
        lambda *_, **__: {"bid": 100.0, "ask": 100.2, "bid_qty": 1000, "ask_qty": 1000},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.resolve_weekly_atm",
        lambda price: {"ce": ("TCE", 50), "pe": ("TPE", 50)},
    )
    monkeypatch.setattr(
        "src.strategies.scalping_strategy.select_strike",
        lambda price, score: type("SI", (), {"strike": int(round(price / 50.0) * 50)})(),
    )

    plan = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    dbg = plan.get("score_dbg")
    assert dbg and "components" in dbg and "final" in dbg
    assert pytest.approx(plan["score"]) == dbg["final"]


def test_shadow_blockers_detects_issues() -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    plan = {
        "micro": {"mode": "SOFT", "spread_pct": 1.2, "cap_pct": 1.0, "depth_ok": False},
        "rr": 1.0,
    }
    shadows = runner._shadow_blockers(plan)
    assert "micro_spread 1.20%>1.00%" in shadows
    assert "micro_depth" in shadows
    assert any(s.startswith("rr_low") for s in shadows)
