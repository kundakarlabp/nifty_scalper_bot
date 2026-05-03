from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'src' / 'nifty_scalper_bot' / 'core' / 'app.py'


def test_app_does_not_call_broker_get_ohlc_directly() -> None:
    text = APP.read_text(encoding='utf-8')
    assert 'ctx.broker_client.get_ohlc' not in text
    assert '.historical_data(' not in text


def test_app_does_not_thread_async_mdm_fetch_history() -> None:
    text = APP.read_text(encoding='utf-8')
    assert 'asyncio.to_thread(\n                    ctx.market_data_manager.fetch_history' not in text
    assert 'asyncio.to_thread(ctx.market_data_manager.fetch_history' not in text
