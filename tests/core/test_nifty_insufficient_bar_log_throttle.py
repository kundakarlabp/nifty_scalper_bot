from nifty_scalper_bot.strategies.runner import StrategyRunner


def test_nifty_insufficient_log_throttle_key_present():
    src = open('src/nifty_scalper_bot/strategies/runner.py', encoding='utf-8').read()
    assert 'runner_eval_insufficient:{symbol}:{stage}:{reason}' in src
    assert '30.0' in src
