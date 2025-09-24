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


def test_score_breakdown_present_on_early_block(
    monkeypatch: pytest.MonkeyPatch, strategy_config: StrategySettings
) -> None:
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
    from src.signals.regime_detector import RegimeResult

    monkeypatch.setattr(
        "src.strategies.scalping_strategy.detect_market_regime",
        lambda **_: RegimeResult("RANGE", 0.0, 0.0, 0.0, 0.0, "test"),
    )

    plan = strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    dbg = plan.get("score_dbg")
    assert dbg is not None
    assert set(dbg["components"]) == {"trend", "momentum", "pullback", "breakout"}
    assert dbg["weights"] == {"trend": 0.4, "momentum": 0.3, "pullback": 0.2, "breakout": 0.1}
    assert dbg["penalties"] == {}
    assert dbg["raw"] >= 0.0
    assert pytest.approx(plan["score"]) == dbg["final"]
    assert isinstance(dbg["threshold"], float)


def test_last_debug_has_score_info_on_early_block(
    monkeypatch: pytest.MonkeyPatch, strategy_config: StrategySettings
) -> None:
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
    from src.signals.regime_detector import RegimeResult

    monkeypatch.setattr(
        "src.strategies.scalping_strategy.detect_market_regime",
        lambda **_: RegimeResult("RANGE", 0.0, 0.0, 0.0, 0.0, "test"),
    )

    strategy.generate_signal(df, current_price=float(df["close"].iloc[-1]))
    score = strategy._last_debug.get("score")
    assert isinstance(score, float) and score >= 0.0
    assert "score_dbg" in strategy._last_debug


def test_score_command_reports_when_breakdown_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from src.notifications.telegram_controller import TelegramController

    runner = SimpleNamespace(
        _score_items={},
        _score_total=0.0,
        strategy_cfg=SimpleNamespace(min_signal_score=0.3),
    )
    monkeypatch.setattr(
        "src.notifications.telegram_controller.StrategyRunner.get_singleton",
        lambda: runner,
    )
    sent: list[str] = []
    tc = TelegramController.__new__(TelegramController)
    tc._allowlist = {1}
    tc._send = lambda text, **_: sent.append(text)
    tc._status_provider = None
    tc._runner_tick = None
    tc._cancel_all = None
    tc._base = ""
    tc._session = SimpleNamespace(post=lambda *a, **k: None)
    tc._timeout = 0

    tc._handle_update({"message": {"chat": {"id": 1}, "text": "/score"}})
    assert sent and "total=0.0" in sent[0]


def test_shadow_blockers_detects_issues() -> None:
    runner = StrategyRunner(telegram_controller=DummyTelegram())
    plan = {
        "micro": {
            "mode": "SOFT",
            "spread_pct": 1.2,
            "spread_cap_pct": 1.0,
            "depth_ok": False,
        },
        "rr": 1.0,
    }
    shadows = runner._shadow_blockers(plan)
    assert "micro_spread 1.20%>1.00%" in shadows
    assert "micro_depth" in shadows
    assert any(s.startswith("rr_low") for s in shadows)
