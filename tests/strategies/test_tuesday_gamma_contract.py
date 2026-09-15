from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from nifty_scalper_bot.strategies.elite_strategies.config_models import (
    TuesdayGammaBuyerStrategyConfig,
)
from nifty_scalper_bot.strategies.elite_tuesday_gamma_buyer import (
    EliteTuesdayGammaBuyer,
)


IST = ZoneInfo("Asia/Kolkata")
SYMBOL_CE = "NFO:NIFTY2691523400CE"
SYMBOL_PE = "NFO:NIFTY2691523400PE"


class _Clock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now_local(self) -> datetime:
        return self._value


def _strategy(now: datetime, *, quantity: int = 1) -> EliteTuesdayGammaBuyer:
    strategy = EliteTuesdayGammaBuyer(
        TuesdayGammaBuyerStrategyConfig(min_confidence=0.0, quantity=quantity),
        indicator_engine=None,
    )
    strategy.clock = _Clock(now)
    return strategy


def _context(side: str) -> dict[str, object]:
    bullish = side == "CE"
    spot = 23450.0 if bullish else 23350.0
    vwap = 23400.0
    ema_fast = 23420.0 if bullish else 23380.0
    ema_slow = 23400.0
    return {
        "direction_bias": side,
        "underlying_direction_bias": side,
        "context_fresh": True,
        "context_age_seconds": 0.1,
        "days_to_expiry": 0,
        "atr": 8.0,
        "spot_context": {
            "symbol": "NSE:NIFTY",
            "role": "spot_context",
            "ltp": spot,
            "close": spot,
            "vwap": vwap,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "atr": 120.0,
        },
        "futures_context": {
            "symbol": "NFO:NIFTY26SEPFUT",
            "role": "futures_context",
            "ltp": spot + 10.0,
            "close": spot + 10.0,
            "vwap": vwap + 10.0,
            "ema_fast": ema_fast + 10.0,
            "ema_slow": ema_slow + 10.0,
            "volume": 1800.0,
            "avg_volume": 1000.0,
            "atr": 125.0,
        },
    }


def test_expiry_gamma_remains_feature_gated(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "directional_scalp")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = _strategy(datetime(2026, 9, 15, 10, 0, tzinfo=IST))

    assert strategy._evaluate_signal(SYMBOL_CE, _context("CE"), 120.0) is None


def test_expiry_gate_applies_even_without_injected_clock(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = EliteTuesdayGammaBuyer(
        TuesdayGammaBuyerStrategyConfig(min_confidence=0.0),
        indicator_engine=None,
    )
    indicators = _context("CE")
    indicators["days_to_expiry"] = 1

    assert strategy._evaluate_signal(SYMBOL_CE, indicators, 120.0) is None


def test_pe_setup_has_positive_quality_and_runner_final_score_contract(monkeypatch) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = _strategy(datetime(2026, 9, 15, 10, 0, tzinfo=IST))

    signal = strategy._evaluate_signal(SYMBOL_PE, _context("PE"), 120.0)

    assert signal is not None
    assert signal.metadata["side"] == "PE"
    assert signal.metadata["strategy_score"] > 0.0
    assert signal.metadata["raw_setup_score"] == signal.metadata["strategy_score"]
    assert signal.metadata["setup_pass"] is True
    assert signal.metadata["preliminary_only"] is True
    assert signal.metadata["requires_runner_final_score"] is True


def test_underlying_context_is_required_and_option_price_never_substitutes_for_spot(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = _strategy(datetime(2026, 9, 15, 10, 0, tzinfo=IST))
    indicators = _context("CE")
    indicators.pop("spot_context")
    indicators.pop("futures_context")

    assert strategy._evaluate_signal(SYMBOL_CE, indicators, 120.0) is None
    assert strategy.last_no_vote_reason == "underlying_context_unavailable"


def test_strategy_requests_nominal_quantity_and_leaves_sizing_to_risk_manager(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STRATEGY_MODE", "expiry_gamma")
    monkeypatch.setenv("ALLOW_EXPIRY_GAMMA_STRATEGIES", "true")
    strategy = _strategy(
        datetime(2026, 9, 15, 10, 0, tzinfo=IST),
        quantity=1,
    )

    signal = strategy._evaluate_signal(SYMBOL_CE, _context("CE"), 120.0)

    assert signal is not None
    assert signal.quantity == 1
    assert signal.metadata["sizing_owner"] == "RiskManager"
    assert not hasattr(strategy, "account_state")
