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
