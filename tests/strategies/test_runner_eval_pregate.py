from __future__ import annotations

import logging
from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


OPTION = "NFO:NIFTY26JUN23800CE"
SPOT = "NSE:NIFTY"
FUT = "NFO:NIFTY26JUNFUT"


class QuoteHub:
    def __init__(self, quotes: dict[str, dict]):
        self.quotes = quotes

    def get_quote(self, symbol: str, allow_pull: bool = False):  # noqa: ARG002
        return self.quotes.get(symbol)

    def quote_update_version(self, symbol: str) -> int:  # noqa: ARG002
        return 7


class EvalCounter:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self) -> None:
        self.calls += 1


def _runner(*, quote: dict | None = None, cfg: StrategyRunnerConfig | None = None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._config = cfg or StrategyRunnerConfig()
    r._data_hub = QuoteHub({OPTION: quote or {"ltp": 100.0, "bid": 99.5, "ask": 100.5}})
    r._market_data = None
    r._last_tick = {}
    r._last_periodic_eval_at_by_symbol = {}
    r._last_eval_bar_key_by_symbol = {}
    r._last_pregate_log_at_by_symbol_reason = {}
    r._quote_update_versions = {OPTION: 7}
    r._candle_versions = {OPTION: 1, SPOT: 1, FUT: 1}
    r._last_bar_ts = {}
    r._data_phase = {OPTION: "LIVE", SPOT: "LIVE", FUT: "LIVE"}
    r._logger = logging.getLogger("pregate-test")
    r._active_selected_ce = OPTION
    r._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    r._active_option_symbols = {OPTION, r._active_selected_pe}
    r._active_basket_token_by_symbol = {OPTION: 123, r._active_selected_pe: 456}
    return r


def _run_eval_if_allowed(runner: StrategyRunner, evaluator: EvalCounter, symbol: str = OPTION, *, bar_key="bar-1") -> str:
    skipped, reason, details = runner._should_skip_symbol_eval(symbol, {"ltp": 100.0}, bar_key=bar_key)
    if skipped:
        return reason
    runner._mark_symbol_eval_allowed(symbol, bar_key=details.get("bar_key"))
    evaluator.evaluate()
    return reason


def test_same_bar_tick_within_5_seconds_does_not_call_strategy_again(monkeypatch) -> None:
    runner = _runner()
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])

    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    clock[0] = 102.0
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "same_bar_periodic_eval_throttled"
    assert evaluator.calls == 1


def test_same_bar_tick_after_5_seconds_allows_periodic_evaluation(monkeypatch) -> None:
    runner = _runner()
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])

    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    clock[0] = 105.1
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    assert evaluator.calls == 2


def test_new_candle_allows_evaluation_immediately(monkeypatch) -> None:
    runner = _runner()
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])

    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    clock[0] = 101.0
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-2") == "ok"
    assert evaluator.calls == 2


def test_option_ltp_below_min_premium_skips_strategy_evaluation() -> None:
    runner = _runner(quote={"ltp": 19.95, "bid": 19.9, "ask": 20.0})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator) == "option_premium_below_min"
    assert evaluator.calls == 0


def test_missing_bid_ask_skips_option_strategy_evaluation() -> None:
    runner = _runner(quote={"ltp": 100.0})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator) == "option_bid_ask_missing_or_invalid"
    assert evaluator.calls == 0


def test_ask_less_than_bid_skips_option_strategy_evaluation() -> None:
    runner = _runner(quote={"ltp": 100.0, "bid": 101.0, "ask": 100.0})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator) == "option_bid_ask_missing_or_invalid"
    assert evaluator.calls == 0


def test_spread_pct_above_threshold_skips_strategy_evaluation() -> None:
    runner = _runner(quote={"ltp": 100.0, "bid": 99.0, "ask": 102.0})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator) == "option_spread_too_wide"
    assert evaluator.calls == 0


def test_valid_premium_bid_ask_and_spread_allows_strategy_evaluation() -> None:
    runner = _runner(quote={"ltp": 100.0, "bid": 99.5, "ask": 100.5})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator) == "ok"
    assert evaluator.calls == 1


def test_premium_and_spread_gates_do_not_apply_to_spot_or_futures_context() -> None:
    runner = _runner(quote={"ltp": 1.0})
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator, SPOT, bar_key="spot-bar") == "ok"
    assert _run_eval_if_allowed(runner, evaluator, FUT, bar_key="fut-bar") == "ok"
    assert evaluator.calls == 2


def test_pregate_skip_does_not_call_strategy_evaluator() -> None:
    runner = _runner(quote={"ltp": 10.0, "bid": 9.9, "ask": 10.1})
    evaluator = EvalCounter()

    reason = _run_eval_if_allowed(runner, evaluator)

    assert reason == "option_premium_below_min"
    assert evaluator.calls == 0


def test_pregate_skip_does_not_affect_hydration_or_active_basket() -> None:
    runner = _runner(quote={"ltp": 10.0, "bid": 9.9, "ask": 10.1})
    before_phase = dict(runner._data_phase)
    before_ce = runner._active_selected_ce
    before_pe = runner._active_selected_pe
    before_tokens = dict(runner._active_basket_token_by_symbol)

    skipped, reason, _details = runner._should_skip_symbol_eval(OPTION, {}, bar_key="bar-1")

    assert skipped is True
    assert reason == "option_premium_below_min"
    assert runner._data_phase == before_phase
    assert runner._active_selected_ce == before_ce
    assert runner._active_selected_pe == before_pe
    assert runner._active_basket_token_by_symbol == before_tokens


def test_pregate_skip_log_includes_event_and_reason(caplog) -> None:
    runner = _runner(quote={"ltp": 10.0, "bid": 9.9, "ask": 10.1})
    skipped, reason, details = runner._should_skip_symbol_eval(OPTION, {}, bar_key="bar-1")

    with caplog.at_level(logging.INFO, logger="pregate-test"):
        runner._log_eval_pregate_skip(OPTION, reason, details)

    assert skipped is True
    record = next(rec for rec in caplog.records if getattr(rec, "event", "") == "RUNNER_EVAL_PREGATE_SKIPPED")
    assert record.reason == "option_premium_below_min"
    assert record.symbol == OPTION
