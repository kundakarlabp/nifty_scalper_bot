from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.market_aware_profit_extension import (
    ContinuationDecision,
    assess_continuation,
    adapt_evaluate_exit_fast,
    extend_final_target_if_supported,
    tighten_market_aware_floor,
)


SYMBOL = "NFO:NIFTY2681124500CE"
FUTURE = "NFO:NIFTY26AUGFUT"


class FakeMarketData:
    def __init__(self, *, strong: bool = True) -> None:
        self.strong = strong
        self.quotes = {
            SYMBOL: {
                "symbol": SYMBOL,
                "ltp": 120.0,
                "bid": 119.8,
                "ask": 120.1,
                "volume": 180_000 if strong else 40_000,
                "oi": 145_000 if strong else 80_000,
                "iv": 0.24 if strong else 0.16,
            },
            "NSE:NIFTY": {
                "symbol": "NSE:NIFTY",
                "ltp": 24_720.0 if strong else 24_610.0,
            },
            FUTURE: {
                "symbol": FUTURE,
                "ltp": 24_755.0 if strong else 24_620.0,
            },
            "NFO:NIFTY2681124500PE": {
                "symbol": "NFO:NIFTY2681124500PE",
                "ltp": 88.0,
                "oi": 210_000 if strong else 70_000,
            },
        }

    def get_quote(self, symbol: str, allow_pull: bool = False):
        del allow_pull
        return self.quotes.get(symbol)

    def get_oi(self, symbol: str):
        quote = self.quotes.get(symbol) or {}
        return quote.get("oi")

    def get_iv(self, symbol: str):
        quote = self.quotes.get(symbol) or {}
        return quote.get("iv")

    def get_greeks(self, symbol: str):
        if symbol == SYMBOL:
            return {"delta": 0.57, "gamma": 0.014, "theta": -0.21}
        return None

    def get_active_contract_basket(self):
        return {
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": FUTURE,
            "selected_ce": SYMBOL,
            "selected_pe": "NFO:NIFTY2681124500PE",
            "option_symbols": [SYMBOL, "NFO:NIFTY2681124500PE"],
        }


class FakeIndicatorEngine:
    def __init__(self, *, strong: bool = True) -> None:
        self.strong = strong

    def get_indicators(self, symbol: str, names):
        del names
        if symbol == SYMBOL:
            return {
                "exchange_vwap": 113.0 if self.strong else 123.0,
                "ema_fast": 116.0 if self.strong else 114.0,
                "ema_slow": 112.0 if self.strong else 118.0,
                "adx": 28.0 if self.strong else 13.0,
                "atr": 3.0,
                "volume": 180_000 if self.strong else 40_000,
                "avg_volume": 90_000,
            }
        if symbol == "NSE:NIFTY":
            return {
                "ltp": 24_720.0 if self.strong else 24_610.0,
                "exchange_vwap": 24_665.0,
                "ema_fast": 24_700.0 if self.strong else 24_635.0,
                "ema_slow": 24_670.0 if self.strong else 24_680.0,
                "adx": 27.0 if self.strong else 14.0,
                "volume": 1_800_000 if self.strong else 700_000,
                "avg_volume": 1_000_000,
                "regime": "TREND_UP" if self.strong else "CHOPPY",
            }
        if symbol == FUTURE:
            return {
                "ltp": 24_755.0 if self.strong else 24_620.0,
                "exchange_vwap": 24_690.0,
                "ema_fast": 24_735.0 if self.strong else 24_640.0,
                "ema_slow": 24_700.0 if self.strong else 24_690.0,
                "adx": 29.0 if self.strong else 13.0,
                "volume": 920_000 if self.strong else 310_000,
                "avg_volume": 500_000,
                "futures_volume_ratio": 1.84 if self.strong else 0.62,
                "regime": "TREND_UP" if self.strong else "CHOPPY",
            }
        return {}


class FakeManager:
    def __init__(self, *, strong: bool = True) -> None:
        self._market_data = FakeMarketData(strong=strong)
        self._indicator_engine = FakeIndicatorEngine(strong=strong)
        self._recent_ticks = {
            SYMBOL: [112.0, 114.0, 116.0, 118.0, 120.0]
            if strong
            else [122.0, 121.5, 121.0, 120.5, 120.0]
        }
        self.saved = 0

    def _breakeven_cost_per_unit(self, bracket):
        del bracket
        return 0.8

    def save_state(self):
        self.saved += 1



def _bracket(*, protected: bool = True, quantity: int = 65):
    return SimpleNamespace(
        bracket_id="entry-1",
        entry_order_id="entry-1",
        trade_lifecycle_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        quantity=quantity,
        remaining_quantity=quantity,
        entry_price=100.0,
        initial_sl_trigger_price=90.0,
        sl_trigger_price=106.0 if protected else 94.0,
        tp_trigger_price=118.0,
        highest_ltp=120.0,
        lowest_ltp=100.0,
        last_ltp=120.0,
        tp_levels=[],
        trailing_config={},
        trade_provenance={
            "profit_extension_entry_oi": 100_000.0,
            "profit_extension_entry_iv": 0.20,
            "profit_extension_entry_volume": 90_000.0,
            "profit_extension_entry_chain_pcr": 1.0,
        },
        updated_at=0.0,
    )


def _final_tp_action():
    return {
        "decision": "EXIT_TARGET",
        "type": "FINAL_TP",
        "price": 120.0,
        "qty": 65,
        "reason": "HARD_TP_BREACH",
    }


def test_strong_continuation_scores_as_extension_candidate(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    decision = assess_continuation(FakeManager(strong=True), _bracket(), 120.0)

    assert isinstance(decision, ContinuationDecision)
    assert decision.extend is True
    assert decision.score >= 5.0
    assert decision.evidence_count >= 4
    assert "premium_momentum" in decision.positive
    assert "underlying_vwap_alignment" in decision.positive


def test_weak_continuation_does_not_extend(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    decision = assess_continuation(FakeManager(strong=False), _bracket(), 120.0)

    assert decision.extend is False
    assert decision.score < 5.0
    assert decision.negative


def test_extension_requires_already_protected_profit_floor(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=False)

    result = extend_final_target_if_supported(
        manager,
        bracket,
        120.0,
        _final_tp_action(),
    )

    assert result is not None
    assert result["type"] == "FINAL_TP"
    assert bracket.tp_trigger_price == 118.0
    assert bracket.sl_trigger_price == 94.0


def test_strong_continuation_extends_target_and_never_weakens_stop(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    monkeypatch.setenv("PROFIT_EXTENSION_STEP_R", "0.75")
    monkeypatch.setenv("PROFIT_EXTENSION_MAX_R", "4.0")
    monkeypatch.setenv("PROFIT_EXTENSION_LOCK_FRACTION", "0.50")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=True)
    old_sl = bracket.sl_trigger_price
    old_tp = bracket.tp_trigger_price

    result = extend_final_target_if_supported(
        manager,
        bracket,
        120.0,
        _final_tp_action(),
    )

    assert result is None
    assert bracket.tp_trigger_price > old_tp
    assert bracket.sl_trigger_price >= old_sl
    assert bracket.sl_trigger_price < 120.0
    assert bracket.trade_provenance["profit_extension_count"] == 1
    assert manager.saved == 1


def test_one_lot_winner_can_extend_without_fractional_tp1(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    manager = FakeManager(strong=True)
    bracket = _bracket(quantity=65)

    result = extend_final_target_if_supported(
        manager,
        bracket,
        120.0,
        _final_tp_action(),
    )

    assert result is None
    assert bracket.tp_levels == []
    assert bracket.remaining_quantity == 65
    assert bracket.tp_trigger_price > 118.0


def test_extension_is_bounded_by_max_r(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "5")
    monkeypatch.setenv("PROFIT_EXTENSION_STEP_R", "1.0")
    monkeypatch.setenv("PROFIT_EXTENSION_MAX_R", "2.25")
    manager = FakeManager(strong=True)
    bracket = _bracket()
    bracket.tp_trigger_price = 121.5

    result = extend_final_target_if_supported(
        manager,
        bracket,
        122.0,
        {**_final_tp_action(), "price": 122.0},
    )

    assert result is None
    assert bracket.tp_trigger_price == pytest.approx(122.5)

    second = extend_final_target_if_supported(
        manager,
        bracket,
        123.0,
        {**_final_tp_action(), "price": 123.0},
    )
    assert second is not None
    assert second["type"] == "FINAL_TP"
    assert bracket.tp_trigger_price == pytest.approx(122.5)


def test_weak_market_can_only_tighten_existing_profit_floor(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_TIGHTEN_SCORE", "-2")
    monkeypatch.setenv("PROFIT_WEAK_LOCK_FRACTION", "0.65")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=True)
    bracket.highest_ltp = 120.0
    old_sl = bracket.sl_trigger_price

    changed = tighten_market_aware_floor(manager, bracket, 120.0)

    assert changed is True
    assert bracket.sl_trigger_price >= old_sl
    assert bracket.sl_trigger_price < 120.0


def test_market_aware_tightening_never_acts_before_one_r(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_TIGHTEN_SCORE", "-2")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=True)
    bracket.highest_ltp = 108.0
    bracket.last_ltp = 108.0
    old_sl = bracket.sl_trigger_price

    changed = tighten_market_aware_floor(manager, bracket, 108.0)

    assert changed is False
    assert bracket.sl_trigger_price == old_sl


def test_hard_stop_and_partial_tp_are_never_overridden(monkeypatch):
    monkeypatch.setenv("MARKET_AWARE_PROFIT_EXTENSION_SCORE", "0")
    manager = FakeManager(strong=True)
    bracket = _bracket()
    calls = []

    def hard_stop(_self, _bracket, _ltp, *, committed_sl=None):
        calls.append(committed_sl)
        return {"decision": "EXIT_STOP", "type": "SL", "reason": "HARD_SL_BREACH"}

    wrapped = adapt_evaluate_exit_fast(hard_stop)
    action = wrapped(manager, bracket, 89.0, committed_sl=90.0)
    assert action is not None and action["type"] == "SL"
    assert calls == [90.0]

    def partial(_self, _bracket, _ltp, *, committed_sl=None):
        del committed_sl
        return {"decision": "EXIT_TARGET", "type": "PARTIAL_TP", "reason": "TP1 Hit"}

    wrapped_partial = adapt_evaluate_exit_fast(partial)
    action = wrapped_partial(manager, bracket, 110.0, committed_sl=106.0)
    assert action is not None and action["type"] == "PARTIAL_TP"


def test_facade_installs_market_aware_extension_once():
    from nifty_scalper_bot.execution.bracket_manager import BracketManager

    assert getattr(
        BracketManager,
        "_market_aware_profit_extension_installed",
        False,
    ) is True
