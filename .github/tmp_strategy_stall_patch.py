from pathlib import Path

runner_path = Path("src/nifty_scalper_bot/strategies/runner.py")
test_path = Path("tests/strategies/test_runner_market_closed_watchdog.py")

runner = runner_path.read_text(encoding="utf-8")
duplicate = '''            # ✅ FIX: Throttle stall warning to 30s — same-bar-skip causes expected
            # gaps between _last_global_eval_ts updates (one eval per bar ≈ 60s cycle).
            # ✅ FIX D: Raise stall threshold to 120s. Options tick once per ~13min;
            # the 5s threshold fired constantly between normal tick batches.
            # A genuine stall = no strategy evaluation for > 2 full minutes with active ticks.
            if (
                self.ready
                and now_mono - self._last_global_eval_ts > 120.0
                and now_mono - self._last_stall_warn_ts > 120.0
            ):
                self._last_stall_warn_ts = now_mono
                if not is_market_open_now():
                    log_throttled(
                        self._logger,
                        "strategy_stall_check_skipped_market_closed",
                        "STALL_CHECK_SKIPPED reason=market_closed",
                        interval_sec=300.0,
                        level=logging.DEBUG,
                        extra={
                            "event": "STALL_CHECK_SKIPPED",
                            "reason": "market_closed",
                        },
                    )
                else:
                    self._logger.warning(
                        "Strategy evaluation stalled >120s (once per 120s)",
                        extra={
                            "event": "strategy_eval_stall",
                            "stall_sec": round(now_mono - self._last_global_eval_ts, 1),
                        },
                    )
'''
if runner.count(duplicate) != 1:
    raise SystemExit("duplicate stall block not found exactly once")
runner = runner.replace(duplicate, "")

old_condition = '''        genuine_stall = (now - self._last_global_eval_ts) > 90.0
        if self.ready and tick_flowing and eval_stalled and genuine_stall:
'''
new_condition = '''        genuine_stall = (now - self._last_global_eval_ts) > 90.0
        risk_breaker_tripped = bool(
            getattr(self._risk_manager, "_breaker_tripped", False)
        )
        if (
            self.ready
            and tick_flowing
            and eval_stalled
            and genuine_stall
            and not risk_breaker_tripped
        ):
'''
if runner.count(old_condition) != 1:
    raise SystemExit("watchdog condition not found exactly once")
runner = runner.replace(old_condition, new_condition)

old_reset = '''        elif self._eval_stall_recovery_attempted and not eval_stalled:
            self._eval_stall_recovery_attempted = False
'''
new_reset = '''        elif self._eval_stall_recovery_attempted and (
            not eval_stalled or risk_breaker_tripped
        ):
            self._eval_stall_recovery_attempted = False
'''
if runner.count(old_reset) != 1:
    raise SystemExit("watchdog recovery reset not found exactly once")
runner = runner.replace(old_reset, new_reset)
runner_path.write_text(runner, encoding="utf-8")

regressions = '''


def test_strategy_stall_warning_has_single_owner():
    source = _read_runner_source()
    assert "Strategy evaluation stalled >120s (once per 120s)" not in source
    assert source.count("Strategy eval genuinely stalled while ticks flowing (>90s)") == 1


def test_strategy_stall_watchdog_skips_active_risk_breaker(monkeypatch):
    import nifty_scalper_bot.strategies.runner as runner_module
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    class _RiskManager:
        _breaker_tripped = True

    class _MarketData:
        def _required_live_symbols(self):
            return set()

    runner = StrategyRunner.__new__(StrategyRunner)
    runner.ready = True
    runner._risk_manager = _RiskManager()
    runner._market_data = _MarketData()
    runner._last_tick_seen_ts = time.monotonic()
    runner._last_global_eval_ts = time.monotonic() - 200.0
    runner._eval_stall_recovery_attempted = False
    runner._candle_engines = {}
    runner._active_symbols = set()
    runner._tracked_symbols = set()
    runner._last_tick_time_by_symbol = {}
    runner._active_option_symbols = set()
    runner._active_selected_ce = None
    runner._active_selected_pe = None
    runner._selected_ce_symbol = None
    runner._selected_pe_symbol = None
    runner._pending_selected_ce = None
    runner._pending_selected_pe = None
    runner._active_contract_basket = None
    runner._data_hub = None
    runner._last_ws_stale_log_ts_by_symbol = {}
    runner._last_ws_reconnect_attempt_ts = 0.0
    runner._log_throttle_state = {}
    runner._logger = logging.getLogger("test.runner.breaker_stall")
    warnings = []
    runner._logger.warning = lambda msg, *args, **kwargs: warnings.append(
        msg % args if args else msg
    )
    recoveries = []
    runner._recover_strategy_eval_stall_once = lambda now: recoveries.append(now)

    monkeypatch.setattr(runner_module, "is_market_open_now", lambda: True)
    runner._health_watchdog()

    assert warnings == []
    assert recoveries == []
    assert runner._eval_stall_recovery_attempted is False
'''
tests = test_path.read_text(encoding="utf-8")
if "def test_strategy_stall_warning_has_single_owner" not in tests:
    test_path.write_text(tests + regressions, encoding="utf-8")
