# tests/test_realtime_trader_smoke.py
import types
import pandas as pd
import numpy as np
import datetime as dt
import builtins

import pytest

# ---- Patch TelegramController to a no-op before creating RealTimeTrader ----
import src.data_streaming.realtime_trader as rtmod


class _NoOpTelegram:
    def __init__(self, *a, **kw):
        pass

    def send_startup_alert(self):  # called on init
        pass

    def start_polling(self):
        pass

    def stop_polling(self):
        pass

    def send_message(self, *a, **kw):
        pass

    def send_realtime_session_alert(self, *a, **kw):
        pass


rtmod.TelegramController = _NoOpTelegram  # ensure no network


# ---- Minimal stub executor used by tests ------------------------------------
class StubExecutor:
    def __init__(self):
        self.kite = types.SimpleNamespace(
            ltp=self._ltp,
        )
        self._ltp_map = {}
        self.placed = []
        self.setup = []
        self.trails = []
        self.cancels = 0
        self.synced = 0
        self._active = {}

    # RealTimeTrader expects these:
    def place_entry_order(self, **kwargs):
        oid = f"O{len(self.placed) + 1}"
        self.placed.append(kwargs)
        # register active
        self._active[oid] = True
        return oid

    def setup_gtt_orders(self, **kwargs):
        self.setup.append(kwargs)
        return True

    def update_trailing_stop(self, order_id, current_price, atr):
        self.trails.append((order_id, current_price, atr))

    def get_active_orders(self):
        # Return dict of active order records; values unused by trader for tests
        return {k: {"dummy": True} for k, v in self._active.items() if v}

    def cancel_all_orders(self):
        self.cancels += 1
        self._active = {}

    def sync_and_enforce_oco(self):
        # For this smoke test, mark all as closed to simulate broker fills
        self.synced += 1
        self._active = {}

    # --- ltp stub ---
    def _ltp(self, symbols):
        # Return last_price for each symbol
        out = {}
        for s in symbols:
            out[s] = {"last_price": self._ltp_map.get(s, 100.0)}
        return out


def _bars(n=60, seed=7):
    """Generate a simple OHLCV DataFrame with monotonic time index."""
    rng = pd.date_range(end=dt.datetime.now(), periods=n, freq="min")
    rs = np.random.RandomState(seed)
    close = np.cumsum(rs.randn(n)) + 100
    high = close + rs.rand(n) * 0.8
    low = close - rs.rand(n) * 0.8
    open_ = close + rs.randn(n) * 0.2
    vol = rs.randint(100, 1000, n)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=rng,
    )
    return df


@pytest.fixture
def trader(monkeypatch):
    # Relax trading-hours guard for tests
    monkeypatch.setattr(rtmod.RealTimeTrader, "ALLOW_OFFHOURS_TESTING", True, raising=False)

    # Create trader and inject stub executor
    t = rtmod.RealTimeTrader()
    t.order_executor = StubExecutor()
    # Avoid scheduler noise in unit tests
    t.is_trading = True
    return t


def test_warmup_filter_blocks_trades(trader, monkeypatch):
    # Force WARMUP_BARS > bars we pass to ensure block
    monkeypatch.setattr(trader, "WARMUP_BARS", 80, raising=False)

    # Build short data for an option symbol
    short_df = _bars(n=30)
    spot_df = _bars(n=30)

    options_data = {"NFO:NIFTY24AUG20000CE": short_df}
    selected = [{"symbol": "NFO:NIFTY24AUG20000CE"}]

    trader._process_selected_strikes(selected, options_data, spot_df)

    assert len(trader.order_executor.placed) == 0, "Trade should not place without warmup bars"


def test_single_position_policy(trader, monkeypatch):
    # Limit to one; pretend we already have one active
    monkeypatch.setattr(trader, "MAX_CONCURRENT_TRADES", 1, raising=False)
    trader.order_executor._active = {"O_EXISTING": True}

    df = _bars(n=100)
    spot_df = _bars(n=100)
    options_data = {"NFO:NIFTY24AUG20000CE": df}
    selected = [{"symbol": "NFO:NIFTY24AUG20000CE"}]

    trader._process_selected_strikes(selected, options_data, spot_df)
    assert len(trader.order_executor.placed) == 0, "Should respect single-position policy"


def test_circuit_breaker_drawdown(trader, monkeypatch):
    # Start day equity and set a big negative PnL
    trader.daily_start_equity = 100000.0
    trader.daily_pnl = -4000.0  # -4%

    monkeypatch.setattr(trader, "MAX_DAILY_DRAWDOWN_PCT", 0.03, raising=False)
    assert trader._is_circuit_breaker_tripped() is True


def test_trailing_worker_calls_executor(trader):
    # Register one open trade to trail
    oid = "O1"
    trader.active_trades[oid] = {
        "order_id": oid,
        "symbol": "NFO:NIFTY24AUG20000CE",
        "direction": "BUY",
        "quantity": 75,
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "target": 110.0,
        "confidence": 9.0,
        "atr": 2.0,
        "status": "OPEN",
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    }
    # Seed LTP for that symbol
    trader.order_executor._ltp_map["NFO:NIFTY24AUG20000CE"] = 104.5

    trader._trailing_tick()
    assert trader.order_executor.trails, "Trailing should have been attempted"


def test_oco_sync_finalizes_trade(trader):
    # Mark an active trade, then sync should close it and finalize PnL
    oid = "O2"
    trader.active_trades[oid] = {
        "order_id": oid,
        "symbol": "NFO:NIFTY24AUG20000PE",
        "direction": "SELL",
        "quantity": 50,
        "entry_price": 120.0,
        "stop_loss": 130.0,
        "target": 100.0,
        "confidence": 8.5,
        "atr": 3.0,
        "status": "OPEN",
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    }
    trader.order_executor._active[oid] = True

    # First tick: executor clears actives
    trader._oco_and_housekeeping_tick()

    # After tick, our trade should be finalized and removed
    assert oid not in trader.active_trades
