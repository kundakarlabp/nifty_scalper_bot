from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import OptionSideReadiness, StrategyRunner


def _runner() -> StrategyRunner:
    runner = object.__new__(StrategyRunner)
    runner._active_selected_ce = "NFO:CE"
    runner._active_selected_pe = "NFO:PE"
    runner._runtime_live_orders_armed = True
    runner._logger = SimpleNamespace(info=lambda *a, **k: None)
    return runner


def _snap(ce_ok: bool, pe_ok: bool):
    return {
        "CE": OptionSideReadiness(
            "CE",
            "NFO:CE",
            1,
            ce_ok,
            ce_ok,
            True,
            50 if ce_ok else 49,
            50,
            ce_ok,
            ce_ok,
            () if ce_ok else ("history_cold",),
        ),
        "PE": OptionSideReadiness(
            "PE",
            "NFO:PE",
            2,
            pe_ok,
            pe_ok,
            True,
            30 if pe_ok else 29,
            30,
            pe_ok,
            pe_ok,
            () if pe_ok else ("quote_missing",),
        ),
    }


def test_readiness_for_candidate_symbol_returns_candidate_side(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        runner, "_option_side_readiness_snapshot", lambda **_: _snap(True, False)
    )
    assert runner._readiness_for_candidate_symbol("NFO:CE").executable is True
    pe = runner._readiness_for_candidate_symbol("NFO:PE")
    assert pe is not None
    assert pe.side == "PE"
    assert pe.executable is False
    assert pe.blockers == ("quote_missing",)


def test_independent_required_bars_are_preserved_in_snapshot(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        runner, "_option_side_readiness_snapshot", lambda **_: _snap(False, True)
    )
    ce = runner._readiness_for_candidate_symbol("NFO:CE")
    pe = runner._readiness_for_candidate_symbol("NFO:PE")
    assert ce.required_bars == 50 and ce.history_count == 49 and not ce.history_ready
    assert pe.required_bars == 30 and pe.history_count == 30 and pe.history_ready


def test_desired_only_subscription_is_not_ready(monkeypatch):
    runner = _runner()
    mdm = SimpleNamespace(
        _token_by_symbol={"NFO:CE": 1, "NFO:PE": 2},
        _desired_tokens={1},
        _subscribed_tokens=set(),
        _confirmed_subscriptions=set(),
        _active_subscribed_symbols=set(),
        _ws=SimpleNamespace(_tokens=set()),
    )
    runner._market_data = mdm
    runner._indicator_engine = SimpleNamespace(get_history=lambda sym: [1] * 50)
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe="NFO:PE"
    )
    runner._get_cached_quote_for_live_entry = lambda sym: {
        "ltp": 100,
        "bid": 99,
        "ask": 101,
        "depth_available": True,
    }
    runner._is_option_symbol_tick_fresh = lambda sym, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda sym: True
    runner._required_bars_for_symbol = lambda sym: 50
    snap = runner._option_side_readiness_snapshot()
    assert snap["CE"].subscription_requested is True
    assert snap["CE"].subscribed is False


def test_current_generation_fresh_tick_can_prove_subscription(monkeypatch):
    runner = _runner()
    mdm = SimpleNamespace(
        _token_by_symbol={"NFO:CE": 1, "NFO:PE": 2},
        _desired_tokens=set(),
        _subscribed_tokens=set(),
        _confirmed_subscriptions=set(),
        _active_subscribed_symbols=set(),
        _ws=SimpleNamespace(_tokens=set()),
        _symbol_subscription_generation={"NFO:CE": 7},
        _symbol_first_tick_generation={"NFO:CE": 7},
    )
    runner._market_data = mdm
    runner._indicator_engine = SimpleNamespace(get_history=lambda sym: [1] * 50)
    runner._current_active_contract_selection = lambda: SimpleNamespace(
        selected_ce="NFO:CE", selected_pe=None
    )
    runner._get_cached_quote_for_live_entry = lambda sym: {
        "ltp": 100,
        "bid": 99,
        "ask": 101,
        "depth_available": True,
    }
    runner._is_option_symbol_tick_fresh = lambda sym, max_age_s=60.0: True
    runner._selected_option_has_real_depth = lambda sym: True
    runner._required_bars_for_symbol = lambda sym: 50
    assert runner._option_side_readiness_snapshot()["CE"].subscribed is True
    mdm._symbol_first_tick_generation["NFO:CE"] = 6
    assert runner._option_side_readiness_snapshot()["CE"].subscribed is False
