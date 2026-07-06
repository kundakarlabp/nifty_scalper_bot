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


def test_mdm_closed_bar_is_single_pipeline_source() -> None:
    """MDM is the single runtime candle builder: its closed bars populate the
    pipeline store via _publish_closed_bar, duplicates dedupe, and neither MDM
    nor the runner feeds pipeline.on_tick live anymore."""
    import inspect
    import logging
    from datetime import datetime, timezone

    from nifty_scalper_bot.data.market_data_manager import MarketDataManager
    from nifty_scalper_bot.data.pipeline import get_pipeline

    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._logger = logging.getLogger("test")
    mdm._bar_subscribers = []
    sym = "NFO:NIFTYTESTPIPE24050CE"
    bar = {
        "symbol": sym,
        "timestamp": datetime(2026, 7, 7, 9, 16, tzinfo=timezone.utc),
        "open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 10,
    }
    mdm._publish_closed_bar(bar)
    mdm._publish_closed_bar(bar)  # duplicate minute dedupes
    assert len(get_pipeline().store.get(sym)) == 1

    # No live per-tick pipeline feed remains in MDM's consumer or the runner.
    import nifty_scalper_bot.strategies.runner as runner_mod
    mdm_src = inspect.getsource(MarketDataManager)
    assert "get_pipeline().on_tick" not in mdm_src
    runner_src = inspect.getsource(runner_mod.StrategyRunner)
    assert "_pipeline.on_tick" not in runner_src
