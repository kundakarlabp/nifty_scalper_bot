from __future__ import annotations

import logging

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


OPTION = "NFO:NIFTY26JUN23800CE"
SPOT = "NSE:NIFTY"
FUT = "NFO:NIFTY26JUNFUT"


class QuoteHub:
    def quote_update_version(self, symbol: str) -> int:  # noqa: ARG002
        return 7


class EvalCounter:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self) -> None:
        self.calls += 1


def _runner(*, cfg: StrategyRunnerConfig | None = None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._config = cfg or StrategyRunnerConfig()
    r._data_hub = QuoteHub()
    r._market_data = None
    r._last_tick = {}
    r._last_periodic_eval_at_by_symbol = {}
    r._last_eval_bar_key_by_symbol = {}
    r._last_pregate_log_at_by_symbol_reason = {}
    r._quote_update_versions = {OPTION: 7}
    r._candle_versions = {OPTION: 1, SPOT: 1, FUT: 1}
    r._last_bar_ts = {}
    r._data_phase = {OPTION: "LIVE", SPOT: "LIVE", FUT: "LIVE"}
    r._logger = logging.getLogger("same-bar-pregate-test")
    r._active_selected_ce = OPTION
    r._active_selected_pe = "NFO:NIFTY26JUN23800PE"
    r._active_option_symbols = {OPTION, r._active_selected_pe}
    r._active_basket_token_by_symbol = {OPTION: 123, r._active_selected_pe: 456}
    r._active_symbols = {OPTION, SPOT, FUT}
    r._subscribed_tokens = {123, 456}
    return r


def _run_eval_if_allowed(
    runner: StrategyRunner,
    evaluator: EvalCounter,
    symbol: str = OPTION,
    *,
    bar_key="bar-1",
) -> str:
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


def test_context_symbols_are_not_blocked_by_option_quality_gates() -> None:
    runner = _runner()
    evaluator = EvalCounter()

    assert _run_eval_if_allowed(runner, evaluator, SPOT, bar_key="spot-bar") == "ok"
    assert _run_eval_if_allowed(runner, evaluator, FUT, bar_key="fut-bar") == "ok"
    assert evaluator.calls == 2


def test_same_bar_skip_does_not_affect_hydration_active_basket_or_subscriptions(monkeypatch) -> None:
    runner = _runner()
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"

    before_phase = dict(runner._data_phase)
    before_ce = runner._active_selected_ce
    before_pe = runner._active_selected_pe
    before_tokens = dict(runner._active_basket_token_by_symbol)
    before_subscribed = set(runner._subscribed_tokens)
    clock[0] = 102.0

    skipped, reason, _details = runner._should_skip_symbol_eval(OPTION, {}, bar_key="bar-1")

    assert skipped is True
    assert reason == "same_bar_periodic_eval_throttled"
    assert runner._data_phase == before_phase
    assert runner._active_selected_ce == before_ce
    assert runner._active_selected_pe == before_pe
    assert runner._active_basket_token_by_symbol == before_tokens
    assert runner._subscribed_tokens == before_subscribed


def test_same_bar_skip_log_includes_bar_key_elapsed_interval_and_quote_version(caplog, monkeypatch) -> None:
    runner = _runner()
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    clock[0] = 102.0
    skipped, reason, details = runner._should_skip_symbol_eval(OPTION, {}, bar_key="bar-1")

    with caplog.at_level(logging.INFO, logger="same-bar-pregate-test"):
        runner._log_eval_pregate_skip(OPTION, reason, details)

    assert skipped is True
    record = next(rec for rec in caplog.records if getattr(rec, "event", "") == "RUNNER_EVAL_PREGATE_SKIPPED")
    assert record.reason == "same_bar_periodic_eval_throttled"
    assert record.bar_key == "bar-1"
    assert record.same_bar_elapsed_s == 2.0
    assert record.same_bar_interval_s == 5.0
    assert record.quote_update_version == 7


def test_runtime_same_bar_interval_has_three_second_lower_bound(monkeypatch) -> None:
    runner = _runner(cfg=StrategyRunnerConfig(same_bar_periodic_eval_seconds=1.0))
    evaluator = EvalCounter()
    clock = [100.0]
    monkeypatch.setattr("nifty_scalper_bot.strategies.runner.time.monotonic", lambda: clock[0])

    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    clock[0] = 102.5
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "same_bar_periodic_eval_throttled"
    clock[0] = 103.1
    assert _run_eval_if_allowed(runner, evaluator, bar_key="bar-1") == "ok"
    assert evaluator.calls == 2
