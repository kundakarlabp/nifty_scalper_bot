from nifty_scalper_bot.core import app


def test_nifty_token_fallback_present():
    # Startup token resolver fallback map should contain NSE:NIFTY spot token.
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert '"NSE:NIFTY": 256265' in src


def test_spot_context_hydration_log_present():
    src = open('src/nifty_scalper_bot/core/app.py', encoding='utf-8').read()
    assert 'SPOT_CONTEXT_HYDRATION_RESULT symbol=NSE:NIFTY' in src
