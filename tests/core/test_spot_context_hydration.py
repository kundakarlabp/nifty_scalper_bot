from nifty_scalper_bot.core import app


def test_nifty_token_fallback_present():
    # Startup token resolver fallback map should contain NSE:NIFTY spot token.
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert '"NSE:NIFTY": 256265' in src


def test_spot_context_hydration_log_present():
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert 'SPOT_CONTEXT_HYDRATION_RESULT symbol=NSE:NIFTY' in src


def test_runner_history_consistency_is_readiness_based():
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert "runner_history_count >= required_bars and mdm_history_count >= required_bars" in src


def test_spot_required_falls_back_to_required_candles():
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert 'getattr(ctx.strategy_runner, "_required_candles", 0)' in src
    assert 'or 50' in src


def test_hydration_diagnostics_use_canonical_symbol():
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert "hydrated_counts[canonical_sym] = count" in src
    assert '"RUNNER_MDM_HYDRATION_SYNC symbol=%s mdm_bars=%d runner_ingested=%d", canonical_sym' in src
    assert '"event": "RUNNER_HISTORY_INGESTED"' in src
    assert '"symbol": canonical_sym' in src


def test_history_readiness_uses_indicator_count_without_reading_missing_history() -> None:
    """Readiness checks must not trigger missing-history logging hot paths."""
    from types import SimpleNamespace
    from nifty_scalper_bot.core.strategy_live_safety import _indicator_bars_available

    class _Indicator:
        def __init__(self, count: int) -> None:
            self.count = count
        def history_count(self, _symbol: str) -> int:
            return self.count
        def get_history(self, _symbol: str) -> list:
            raise AssertionError("history copy is unnecessary for bar counts")

    manager = SimpleNamespace(_indicator_engine=_Indicator(0))
    assert _indicator_bars_available(manager, "NSE:NIFTY") == 0
    manager._indicator_engine.count = 74
    assert _indicator_bars_available(manager, "NSE:NIFTY") == 74
