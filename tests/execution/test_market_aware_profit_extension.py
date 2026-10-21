from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import (
    market_aware_profit_extension as profit_extension,
)
from nifty_scalper_bot.execution.market_aware_profit_extension import (
    ContinuationDecision,
    assess_continuation,
    extend_final_target_if_supported,
    tighten_market_aware_floor,
)
from nifty_scalper_bot.execution.ownership import BoundBracketManager
from nifty_scalper_bot.execution.runtime_bracket_manager import RuntimeBracketManager

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
            SYMBOL: (
                [112.0, 114.0, 116.0, 118.0, 120.0]
                if strong
                else [122.0, 121.5, 121.0, 120.5, 120.0]
            )
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


def test_strong_continuation_meets_structural_extension_contract(monkeypatch):
    decision = assess_continuation(FakeManager(strong=True), _bracket(), 120.0)

    assert isinstance(decision, ContinuationDecision)
    assert decision.extend is True
    assert decision.critical_block is False
    assert decision.evidence_count >= 4
    assert "premium_momentum" in decision.positive
    assert "underlying_vwap_alignment" in decision.positive


def test_weak_continuation_does_not_extend(monkeypatch):
    decision = assess_continuation(FakeManager(strong=False), _bracket(), 120.0)

    assert decision.extend is False
    assert decision.extend is False
    assert decision.negative


def test_extension_requires_already_protected_profit_floor(monkeypatch):
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
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=True)
    bracket.highest_ltp = 108.0
    bracket.last_ltp = 108.0
    old_sl = bracket.sl_trigger_price

    changed = tighten_market_aware_floor(manager, bracket, 108.0)

    assert changed is False
    assert bracket.sl_trigger_price == old_sl


def test_native_exit_hook_never_overrides_hard_stop_or_partial_tp(monkeypatch):
    manager = object.__new__(BoundBracketManager)
    bracket = _bracket()
    calls = []

    def hard_stop(_self, _bracket, _ltp, *, committed_sl=None):
        calls.append(committed_sl)
        return {"decision": "EXIT_STOP", "type": "SL", "reason": "HARD_SL_BREACH"}

    monkeypatch.setattr(RuntimeBracketManager, "_evaluate_exit_fast", hard_stop)
    action = BoundBracketManager._evaluate_exit_fast(
        manager,
        bracket,
        89.0,
        committed_sl=90.0,
    )
    assert action is not None and action["type"] == "SL"
    assert calls == [90.0]

    def partial(_self, _bracket, _ltp, *, committed_sl=None):
        del committed_sl
        return {"decision": "EXIT_TARGET", "type": "PARTIAL_TP", "reason": "TP1 Hit"}

    monkeypatch.setattr(RuntimeBracketManager, "_evaluate_exit_fast", partial)
    action = BoundBracketManager._evaluate_exit_fast(
        manager,
        bracket,
        110.0,
        committed_sl=106.0,
    )
    assert action is not None and action["type"] == "PARTIAL_TP"


def test_native_fill_hook_captures_baseline_after_confirmed_activation(monkeypatch):
    manager = object.__new__(BoundBracketManager)
    bracket = _bracket()
    bracket.entry_confirmed = True
    calls = []

    monkeypatch.setattr(
        RuntimeBracketManager,
        "confirm_entry_fill",
        lambda _self, _order_id, _fill_price, _filled_qty=None: "parent-result",
    )
    monkeypatch.setattr(
        BoundBracketManager,
        "get_bracket",
        lambda _self, _order_id: bracket,
    )
    monkeypatch.setattr(
        BoundBracketManager,
        "save_state",
        lambda _self: calls.append("saved"),
    )
    monkeypatch.setattr(
        profit_extension,
        "capture_entry_market_baseline",
        lambda _manager, _bracket: calls.append("captured") or True,
    )

    result = BoundBracketManager.confirm_entry_fill(manager, "entry-1", 100.0, 65)

    assert result == "parent-result"
    assert calls == ["captured", "saved"]


def test_market_aware_hooks_are_owned_explicitly_by_bound_manager():
    assert "_evaluate_exit_fast" in BoundBracketManager.__dict__
    assert "_apply_trailing_math" in BoundBracketManager.__dict__
    assert "confirm_entry_fill" in BoundBracketManager.__dict__
    assert not hasattr(BoundBracketManager, "_market_aware_profit_extension_installed")


def test_market_aware_profit_floor_reprices_exchange_stop() -> None:
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=True)
    bracket.highest_ltp = 120.0
    calls = []
    manager._sync_exchange_protective_stop = (
        lambda target: calls.append(target.sl_trigger_price) or True
    )

    assert tighten_market_aware_floor(manager, bracket, 120.0) is True

    assert calls == [bracket.sl_trigger_price]


def test_target_extension_requires_broker_profit_stop() -> None:
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=True)
    bracket.exchange_stop_order_id = "broker-stop-1"
    bracket.exchange_stop_trigger_price = 94.0
    before = bracket.tp_trigger_price

    result = extend_final_target_if_supported(
        manager, bracket, 120.0, _final_tp_action()
    )

    assert result is not None
    assert result["type"] == "FINAL_TP"
    assert bracket.tp_trigger_price == before


def test_target_extension_accepts_matching_broker_profit_protection() -> None:
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=True)
    bracket.exchange_stop_order_id = "broker-stop-1"
    bracket.exchange_stop_trigger_price = 106.0

    result = extend_final_target_if_supported(
        manager, bracket, 120.0, _final_tp_action()
    )

    assert result is None
    assert bracket.tp_trigger_price > 118.0


def _adaptive_market(manager, *, bid=115.0, quantity=65, spread=0.2, age=0.0):
    from time import time

    quote = manager._market_data.quotes[SYMBOL]
    quote.update(
        {
            "bid": bid,
            "ask": bid + spread,
            "ltp": bid + 0.05,
            "depth": {"buy": [{"price": bid, "quantity": quantity}]},
            "received_at": time() - age,
        }
    )
    return quote


def test_adaptive_lock_opt_in_preserves_existing_trailing(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_PROFIT_LOCK_ENABLED", raising=False)
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=False)
    _adaptive_market(manager)
    before = bracket.sl_trigger_price

    assert profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115) is False
    assert bracket.sl_trigger_price == before


def test_adaptive_lock_protects_cost_and_fixed_points_before_one_r(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    monkeypatch.setenv("ADAPTIVE_PROFIT_MIN_POINTS", "1.0")
    monkeypatch.setenv("ADAPTIVE_PROFIT_SLIPPAGE_POINTS", "0.5")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager, bid=107.0)

    assert profit_extension.tighten_adaptive_profit_lock(manager, bracket, 107)
    assert bracket.sl_trigger_price >= 102.3
    assert bracket.sl_trigger_price < quote["bid"]
    assert bracket.sl_trigger_price > 94


def test_adaptive_lock_uses_independent_continuation_families(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    strong = FakeManager(strong=True)
    weak = FakeManager(strong=False)
    strong_bracket = _bracket(protected=False)
    weak_bracket = _bracket(protected=False)
    _adaptive_market(strong, bid=115)
    weak_quote = _adaptive_market(weak, bid=115)

    assert profit_extension.tighten_adaptive_profit_lock(strong, strong_bracket, 115)
    profit_extension.tighten_adaptive_profit_lock(weak, weak_bracket, 115)
    weak_quote["received_at"] += 0.1
    profit_extension.tighten_adaptive_profit_lock(weak, weak_bracket, 115)
    assert weak_bracket.sl_trigger_price > strong_bracket.sl_trigger_price
    assert weak_bracket.sl_trigger_price <= 114.5


@pytest.mark.parametrize("invalid", ["stale", "missing_depth", "small_qty", "wide"])
def test_adaptive_lock_skips_unexecutable_or_stale_quotes(monkeypatch, invalid):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager)
    if invalid == "stale":
        quote["received_at"] -= 25
    elif invalid == "missing_depth":
        quote.pop("depth")
    elif invalid == "small_qty":
        quote["depth"]["buy"][0]["quantity"] = 20
    elif invalid == "wide":
        quote["ask"] = quote["bid"] + 5

    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price == 94.0


def test_adaptive_lock_never_reuses_one_quote_as_multiple_confirmations(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager, bid=115)
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    first_stop = bracket.sl_trigger_price
    assert profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115) is False
    assert bracket.sl_trigger_price == first_stop

    quote["received_at"] += 0.1
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price >= first_stop


def test_adaptive_lock_defers_when_exchange_stop_is_unverified(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=False)
    bracket.exchange_stop_degraded_reason = (
        "broker_stop_status_unverified:MODIFY PENDING"
    )
    _adaptive_market(manager)
    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price == 94.0


def test_adaptive_lock_respects_regime_and_original_entry_context(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    baseline = _bracket(protected=False)
    degraded = _bracket(protected=False)
    degraded.trade_provenance["direction_contract"] = {"passed": False}
    _adaptive_market(manager)
    profit_extension.tighten_adaptive_profit_lock(manager, baseline, 115)
    profit_extension.tighten_adaptive_profit_lock(manager, degraded, 115)
    quote = manager._market_data.quotes[SYMBOL]
    quote["received_at"] += 0.1
    profit_extension.tighten_adaptive_profit_lock(manager, degraded, 115)
    assert degraded.sl_trigger_price > baseline.sl_trigger_price


def test_adaptive_live_mode_requires_existing_exchange_protection(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    manager._is_live_execution = lambda: True
    bracket = _bracket(protected=False)
    _adaptive_market(manager)

    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price == 94.0

    bracket.exchange_stop_order_id = "protected-stop-1"
    assert profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price > 94.0
    assert "virtual_stop" in bracket.trade_provenance["adaptive_profit_lock"]


def test_adaptive_ignores_disabled_trailing_and_missing_initial_risk(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    _adaptive_market(manager)
    disabled = _bracket(protected=False)
    disabled.trailing_enabled = False
    assert not profit_extension.tighten_adaptive_profit_lock(manager, disabled, 115)
    assert disabled.sl_trigger_price == 94.0

    missing = _bracket(protected=False)
    missing.initial_sl_trigger_price = 0
    assert not profit_extension.tighten_adaptive_profit_lock(manager, missing, 115)
    assert missing.sl_trigger_price == 94.0


def test_adaptive_rejects_contradictory_top_of_book_quote(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager)
    quote["bid"] = 140.0
    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price == 94.0


def test_adaptive_requires_verified_positive_cost_evidence(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    manager._breakeven_cost_per_unit = lambda _bracket: 0.0
    bracket = _bracket(protected=False)
    _adaptive_market(manager, bid=115)
    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert bracket.sl_trigger_price == 94.0


def test_adaptive_atr_does_not_silently_override_strong_giveback(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=False)
    _adaptive_market(manager, bid=115)
    strong = ContinuationDecision(
        extend=True,
        evidence_count=5,
        positive=(
            "premium_momentum", "underlying_vwap_alignment",
            "futures_volume_confirmation",
        ),
        negative=(),
        critical_block=False,
        snapshot={"regime": "TREND_UP"},
    )
    monkeypatch.setattr(profit_extension, "assess_continuation", lambda *_: strong)
    quote = manager._market_data.quotes[SYMBOL]
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    quote["received_at"] += 0.1
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    # 40% of a 15-point MFE = 6 points. ATR=3 must widen, not
    # silently convert it into a 60%-plus lock via high - ATR.
    assert bracket.sl_trigger_price <= 106.5


def test_adaptive_does_not_confirm_state_from_cached_market_assessment(
    monkeypatch,
):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    manager = FakeManager(strong=True)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager, bid=115)
    evaluations = []
    normal = ContinuationDecision(
        extend=False,
        evidence_count=4,
        positive=(),
        negative=(),
        critical_block=False,
        snapshot={"regime": "UNKNOWN"},
    )
    monkeypatch.setattr(
        profit_extension,
        "assess_continuation",
        lambda *_: evaluations.append(1) or normal,
    )
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert len(evaluations) == 1
    assert bracket.trailing_config["_adaptive_profit_pending_count"] == 1
    assert not profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert len(evaluations) == 1
    quote["received_at"] += 0.1
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    assert len(evaluations) == 2


def test_adaptive_peak_age_requires_deterioration_before_tightening(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_PROFIT_LOCK_ENABLED", "true")
    monkeypatch.setenv("ADAPTIVE_PROFIT_STALL_SECONDS", "8")
    manager = FakeManager(strong=False)
    bracket = _bracket(protected=False)
    quote = _adaptive_market(manager, bid=115)
    clock = [1000.0]
    monkeypatch.setattr(profit_extension.time, "monotonic", lambda: clock[0])
    weakening = ContinuationDecision(
        extend=False,
        evidence_count=5,
        positive=(),
        negative=("premium_momentum_reversal", "futures_volume_fade"),
        critical_block=False,
        snapshot={"regime": "RANGE"},
    )
    monkeypatch.setattr(profit_extension, "assess_continuation", lambda *_: weakening)
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 115)
    initial = bracket.sl_trigger_price
    quote["bid"] = 113.5
    quote["ask"] = 113.7
    quote["ltp"] = 113.55
    quote["depth"]["buy"][0]["price"] = 113.5
    quote["received_at"] += 0.1
    clock[0] += 9
    profit_extension.tighten_adaptive_profit_lock(manager, bracket, 113.55)
    assert bracket.trailing_config["_adaptive_profit_peak_age_seconds"] >= 8
    assert bracket.sl_trigger_price > initial
    assert bracket.sl_trigger_price < 113.5


def test_single_exchange_stop_sync_after_all_trail_proposals(monkeypatch):
    from types import SimpleNamespace as Namespace
    import threading

    manager = object.__new__(BoundBracketManager)
    manager._trail_sync_context = threading.local()
    bracket = Namespace(
        symbol=SYMBOL, last_ltp=115.0, sl_trigger_price=101.0,
    )
    calls = []

    def canonical(owner, target):
        target.sl_trigger_price = 103
        owner._sync_exchange_protective_stop(target)
        return True

    def market(owner, target, _ltp):
        target.sl_trigger_price = 105
        owner._sync_exchange_protective_stop(target)
        return True

    def adaptive(owner, target, _ltp):
        target.sl_trigger_price = 108
        owner._sync_exchange_protective_stop(target)
        return True

    monkeypatch.setattr(RuntimeBracketManager, "_apply_trailing_math", canonical)
    monkeypatch.setattr(
        RuntimeBracketManager,
        "_sync_exchange_protective_stop",
        lambda _self, target: calls.append(target.sl_trigger_price) or True,
    )
    monkeypatch.setattr(profit_extension, "tighten_market_aware_floor", market)
    monkeypatch.setattr(profit_extension, "tighten_adaptive_profit_lock", adaptive)
    assert manager._apply_trailing_math(bracket)
    assert calls == [108]


def test_exchange_sync_coalescer_is_thread_scoped(monkeypatch):
    from types import SimpleNamespace as Namespace
    import threading

    manager = object.__new__(BoundBracketManager)
    manager._trail_sync_context = threading.local()
    bracket = Namespace(symbol=SYMBOL, last_ltp=115.0, sl_trigger_price=101.0)
    other = Namespace(symbol=SYMBOL, last_ltp=115.0, sl_trigger_price=90.0)
    calls = []
    monkeypatch.setattr(
        RuntimeBracketManager,
        "_sync_exchange_protective_stop",
        lambda _self, target: calls.append(target.sl_trigger_price) or True,
    )

    def canonical(owner, target):
        owner._sync_exchange_protective_stop(target)
        owner._sync_exchange_protective_stop(other)
        return True

    monkeypatch.setattr(RuntimeBracketManager, "_apply_trailing_math", canonical)
    monkeypatch.setattr(
        profit_extension, "tighten_market_aware_floor", lambda *_: False
    )
    monkeypatch.setattr(
        profit_extension, "tighten_adaptive_profit_lock", lambda *_: False
    )
    assert manager._apply_trailing_math(bracket)
    assert calls == [90.0, 101.0]
