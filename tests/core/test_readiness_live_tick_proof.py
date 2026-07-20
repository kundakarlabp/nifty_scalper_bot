from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app


class Snap:
    def __init__(
        self,
        ltp: float,
        age: float,
        tradable: bool = False,
        bid: float | None = None,
        ask: float | None = None,
        depth: bool = False,
    ):
        self.ltp = ltp
        self.tick_age_s = age
        self.tradable_quote = tradable
        self.bid = bid
        self.ask = ask
        self.depth_available = depth


class Runner:
    def __init__(self) -> None:
        self.calls = []
        self._running = True
        self._indicator_engine = SimpleNamespace(get_history=lambda s: _history_bars())

    def set_runtime_readiness(self, **kwargs):
        self.calls.append(kwargs)


def make_ctx(mdm, live: bool = True):
    if not hasattr(mdm, "hydrate_active_contract_basket"):
        mdm.hydrate_active_contract_basket = lambda basket=None: {
            "hard_ready": True,
            "missing": [],
            "symbols": {},
        }
    return SimpleNamespace(
        settings=SimpleNamespace(execution_mode="LIVE" if live else "PAPER"),
        market_data_manager=mdm,
        strategy_runner=Runner(),
        broker_client=object(),
        order_manager=object(),
        active_trading_universe={
            "spot_symbol": "NSE:NIFTY",
            "selected_ce": "NFO:CE",
            "selected_pe": "NFO:PE",
            "option_symbols": ["NFO:CE", "NFO:PE"],
            "atm_strike": 23400,
        },
        execution_locked_symbols=set(),
        execution_lock_timestamps={},
    )


def _history_bars(count: int = 60):
    end = datetime.now(UTC).replace(second=0, microsecond=0)
    return [
        {
            "timestamp": end - timedelta(minutes=count - idx),
            "open": 1,
            "high": 1,
            "low": 1,
            "close": 1,
            "volume": 1,
        }
        for idx in range(count)
    ]


def _fresh_tick_ts() -> dict[str, str]:
    now_iso = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {"NFO:CE": now_iso, "NFO:PE": now_iso}


@pytest.mark.asyncio
async def test_runtime_readiness_uses_live_tick_without_subscription() -> None:
    snaps = {
        "NSE:NIFTY": Snap(25000, 2),
        "NFO:CE": Snap(100, 2),
        "NFO:PE": Snap(110, 3),
    }
    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=lambda s, **k: _history_bars(),
        _confirmed_subscriptions=set(),
        _symbol_to_token={"NFO:CE": 1, "NFO:PE": 2},
        _last_tick_ts=_fresh_tick_ts(),
    )
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason="test")
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.strategy_runner.calls[-1]["data_hard_ready"] is True
    assert ctx.strategy_runner.calls[-1]["evaluation_ready"] is True
    assert (
        str(ctx.live_block_reason).startswith("execution_not_armed:")
        or ctx.live_orders_armed
    )


@pytest.mark.asyncio
async def test_live_orders_not_armed_without_tradable_quote() -> None:
    snaps = {
        "NSE:NIFTY": Snap(25000, 2),
        "NFO:CE": Snap(100, 2),
        "NFO:PE": Snap(110, 2),
    }
    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=lambda s, **k: _history_bars(),
        _confirmed_subscriptions=set(),
        _symbol_to_token={"NFO:CE": 1, "NFO:PE": 2},
        _last_tick_ts=_fresh_tick_ts(),
    )
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason="test")
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is False
    assert str(ctx.live_block_reason).startswith("execution_not_armed:")


@pytest.mark.asyncio
async def test_live_orders_armed_with_fresh_ltp_bars_and_tradable_depth(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    snaps = {
        "NSE:NIFTY": Snap(25000, 2, True, 24999, 25001, True),
        "NFO:CE": Snap(100, 2, True, 99.95, 100.05, True),
        "NFO:PE": Snap(110, 2, True, 109.95, 110.05, True),
    }
    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=lambda s, **k: _history_bars(),
        has_ws_tradable_quote=lambda s: True,
        _confirmed_subscriptions=set(),
        _symbol_to_token={"NFO:CE": 1, "NFO:PE": 2},
        _last_tick_ts=_fresh_tick_ts(),
    )
    ctx = make_ctx(mdm)
    await app._recompute_and_push_runtime_readiness(ctx, reason="test")
    assert ctx.data_hard_ready is True
    assert ctx.evaluation_ready is True
    assert ctx.live_orders_armed is True
    assert ctx.trading_ready is True


@pytest.mark.asyncio
async def test_ensure_selected_options_hydrated_handles_runner_reseed_failure() -> None:
    symbol = "NFO:CE"

    class RunnerFailing:
        def __init__(self) -> None:
            self._indicator_engine = SimpleNamespace(
                get_history=lambda _s: list(range(5))
            )

        def reseed_history_from_bars(self, *args, **kwargs):
            raise RuntimeError("boom")

    class MdmStub:
        async def hydrate_symbol_history(self, *args, **kwargs) -> None:
            return None

        def get_ohlc_bars(self, _sym: str, limit: int | None = None):
            bars = [
                {
                    "start": "2026-05-01T03:45:00+00:00",
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100.5,
                },
            ]
            return bars[: limit or len(bars)]

    ctx = SimpleNamespace(
        market_data_manager=MdmStub(), strategy_runner=RunnerFailing()
    )

    result = await app._ensure_selected_options_hydrated(ctx, symbol, None, 2, "test")
    assert result[symbol]["ready"] is False


@pytest.mark.asyncio
async def test_live_readiness_treats_depth_only_quote_as_tradable(monkeypatch) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)

    class DepthSnap(Snap):
        def __init__(self, ltp: float, age: float, bid: float, ask: float):
            super().__init__(ltp, age, False, None, None, True)
            self.depth = {
                "buy": [{"price": bid, "quantity": 100}],
                "sell": [{"price": ask, "quantity": 100}],
            }

    snaps = {
        "NSE:NIFTY": Snap(25000, 2),
        "NFO:CE": DepthSnap(100, 2, 99.95, 100.05),
        "NFO:PE": DepthSnap(110, 2, 109.95, 110.05),
    }
    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=lambda s, **k: _history_bars(),
        has_ws_tradable_quote=lambda symbols: True,
        _confirmed_subscriptions={1, 2},
        _symbol_to_token={"NFO:CE": 1, "NFO:PE": 2},
        _last_tick_ts=_fresh_tick_ts(),
    )
    ctx = make_ctx(mdm)

    await app._recompute_and_push_runtime_readiness(ctx, reason="depth-only-test")

    assert ctx.live_orders_armed is True
    assert ctx.selected_ce_exec_ready is True
    assert ctx.selected_pe_exec_ready is True


@pytest.mark.asyncio
async def test_insufficient_futures_context_blocks_when_full_hydration_required(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    snaps = {
        "NSE:NIFTY": Snap(25000, 2),
        "NFO:CE": Snap(100, 2, True, 99.95, 100.05, True),
        "NFO:PE": Snap(110, 2, True, 109.95, 110.05, True),
    }

    def bars(symbol: str, **_kwargs):
        if symbol == "NFO:NIFTY26JUNFUT":
            return []
        return _history_bars()

    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=bars,
        has_ws_tradable_quote=lambda symbols: True,
        _confirmed_subscriptions={1, 2},
        _symbol_to_token={"NFO:CE": 1, "NFO:PE": 2},
        _last_tick_ts={
            **_fresh_tick_ts(),
            "NFO:NIFTY26JUNFUT": (datetime.now(UTC) - timedelta(seconds=999))
            .isoformat()
            .replace("+00:00", "Z"),
        },
    )
    ctx = make_ctx(mdm)
    ctx.active_trading_universe["futures_symbol"] = "NFO:NIFTY26JUNFUT"

    await app._recompute_and_push_runtime_readiness(ctx, reason="futures-soft-test")

    assert ctx.context_exec_ready is False
    assert ctx.live_orders_armed is False
    assert "futures_history_short" in str(ctx.live_block_reason)


@pytest.mark.asyncio
async def test_futures_readiness_blocker_names_live_stale_vs_history_short(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app, "get_market_state", lambda: app.MarketState.OPEN)
    statuses = {
        "NSE:NIFTY": SimpleNamespace(
            role="spot",
            ready_for_evaluation=True,
            ready_for_execution=True,
            blocker_reasons=[],
            live_tick_fresh=True,
            tradable_quote=False,
            depth_available=False,
            mdm_bars=60,
            runner_bars=60,
            indicator_bars=60,
            required_bars=20,
            latest_bar_fresh=True,
        ),
        "NFO:NIFTY26JUNFUT": SimpleNamespace(
            role="futures_context",
            ready_for_evaluation=True,
            ready_for_execution=False,
            blocker_reasons=[],
            live_tick_fresh=False,
            tradable_quote=False,
            depth_available=False,
            mdm_bars=60,
            runner_bars=60,
            indicator_bars=60,
            required_bars=20,
            latest_bar_fresh=True,
        ),
        "NFO:CE": SimpleNamespace(
            role="selected_ce",
            ready_for_evaluation=True,
            ready_for_execution=True,
            blocker_reasons=[],
            live_tick_fresh=True,
            tradable_quote=True,
            depth_available=True,
            mdm_bars=60,
            runner_bars=60,
            indicator_bars=60,
        ),
        "NFO:PE": SimpleNamespace(
            role="selected_pe",
            ready_for_evaluation=True,
            ready_for_execution=True,
            blocker_reasons=[],
            live_tick_fresh=True,
            tradable_quote=True,
            depth_available=True,
            mdm_bars=60,
            runner_bars=60,
            indicator_bars=60,
        ),
    }
    for status in statuses.values():
        status.datahub_bars = getattr(status, "datahub_bars", status.mdm_bars)
        status.bid = getattr(status, "bid", 1.0)
        status.ask = getattr(status, "ask", 1.1)
        status.spread_pct = getattr(status, "spread_pct", 0.1)
    monkeypatch.setattr(app, "_hydration_status_map", lambda *_a, **_k: statuses)
    monkeypatch.setattr(
        app,
        "_status_for_role",
        lambda values, role: next(
            (status for status in values.values() if status.role == role), None
        ),
    )
    snaps = {
        "NSE:NIFTY": Snap(25000, 2),
        "NFO:NIFTY26JUNFUT": Snap(25100, 999),
        "NFO:CE": Snap(100, 2, True, 99.95, 100.05, True),
        "NFO:PE": Snap(110, 2, True, 109.95, 110.05, True),
    }
    mdm = SimpleNamespace(
        get_symbol_snapshot=lambda s: snaps.get(s),
        get_ohlc_bars=lambda s, **_k: _history_bars(),
        has_ws_tradable_quote=lambda symbols: True,
        _confirmed_subscriptions={1, 2, 4},
        _symbol_to_token={
            "NFO:CE": 1,
            "NFO:PE": 2,
            "NFO:NIFTY26JUNFUT": 4,
        },
        _last_tick_ts={
            **_fresh_tick_ts(),
            "NFO:NIFTY26JUNFUT": (datetime.now(UTC) - timedelta(seconds=999))
            .isoformat()
            .replace("+00:00", "Z"),
        },
    )
    ctx = make_ctx(mdm)
    ctx.active_trading_universe["futures_symbol"] = "NFO:NIFTY26JUNFUT"
    ctx.active_contract_basket = dict(ctx.active_trading_universe)

    await app._recompute_and_push_runtime_readiness(ctx, reason="futures-live-stale")

    assert ctx.context_exec_ready is False
    assert "futures_live_tick_stale" in str(ctx.live_block_reason)
