from pathlib import Path

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_duplicate_tick_is_dropped() -> None:
    manager = MarketDataManager()
    tick = {'timestamp': 100.0, 'ltp': 200.0}
    assert manager._is_duplicate('NSE:NIFTY', tick) is False
    assert manager._is_duplicate('NSE:NIFTY', tick) is True


def test_market_data_manager_avoids_missing_tick_instrument_attribute() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "src/nifty_scalper_bot/data/market_data_manager.py"
    ).read_text(encoding="utf-8")
    assert "tick.instrument_token" not in source


def test_mdm_readiness_snapshot_compatibility_alias() -> None:
    manager = MarketDataManager.__new__(MarketDataManager)
    manager._last_readiness_state = {
        "hard_ready": True,
        "spot_ready": True,
        "missing_hard": [],
    }
    assert manager.readiness_snapshot() == manager.readiness_state_snapshot()
