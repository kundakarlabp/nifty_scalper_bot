from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.core import app


class _Mdm:
    def __init__(self, cached_ltp: float = 0.0, hydration: dict[str, Any] | None = None) -> None:
        self.cached_ltp = cached_ltp
        self.calls: list[tuple[str, Any]] = []
        self.desired: set[int] = set()
        self.hydration = hydration or {"hard_ready": False, "missing": ["NFO:CE:quote_missing"], "symbols": {}}

    async def wait_for_fresh_spot_tick(self, *_a: Any, **_k: Any) -> None:
        return None

    def get_cached_ltp(self, *_a: Any, **_k: Any) -> float:
        return self.cached_ltp

    def set_active_contract_basket(self, basket: dict[str, Any]) -> None:
        self.calls.append(("set_active_contract_basket", basket))
        self.desired.update(int(t) for t in basket.get("all_tokens", []) if t)

    def hydrate_active_contract_basket(self, basket: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(("hydrate_active_contract_basket", basket))
        return self.hydration

    def register_symbol(self, symbol: str, token: int) -> None:
        self.calls.append(("register_symbol", (symbol, token)))

    def request_token_subscription(self, token: int, *, symbol: str | None = None) -> bool:
        self.calls.append(("request_token_subscription", (symbol, token)))
        self.desired.add(int(token))
        return True

    def desired_token_count(self) -> int:
        return len(self.desired)

    def ws_token_count(self) -> int:
        return len(self.desired)

    def get_hydration_report(self) -> dict[str, Any]:
        return self.hydration

    def get_symbol_snapshot(self, symbol: str) -> SimpleNamespace:
        return SimpleNamespace(ltp=25000.0 if symbol == "NSE:NIFTY" else 100.0, tick_age_s=0.1, bid=99.0, ask=101.0, tradable_quote=True, depth_available=True)

    def has_ws_tradable_quote(self, _symbols: Any) -> bool:
        return True

    def get_ohlc_bars(self, _symbol: str) -> list[dict[str, float]]:
        return [{"close": 100.0}] * 40


class _Runner:
    def __init__(self) -> None:
        self._active_symbols: set[str] = set()
        self._running = False
        self.symbols: list[str] = []
        self.universe: dict[str, Any] | None = None
        self.readiness: dict[str, Any] | None = None

    def add_symbol(self, symbol: str) -> None:
        self.symbols.append(symbol)
        self._active_symbols.add(symbol)

    def set_active_trading_universe(self, universe: dict[str, Any]) -> None:
        self.universe = universe

    def set_runtime_readiness(self, **kwargs: Any) -> None:
        self.readiness = kwargs

    def on_datahub_tick(self, *_a: Any, **_k: Any) -> None:
        return None


def _basket() -> dict[str, Any]:
    return {
        "spot_symbol": "NSE:NIFTY",
        "spot_token": 256265,
        "futures_symbol": "NFO:NIFTY26JUNFUT",
        "futures_token": 1001,
        "selected_ce": "NFO:NIFTY26JUN25000CE",
        "selected_pe": "NFO:NIFTY26JUN25000PE",
        "atm_ce": "NFO:NIFTY26JUN25000CE",
        "atm_pe": "NFO:NIFTY26JUN25000PE",
        "option_symbols": ["NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
        "ce_symbols": ["NFO:NIFTY26JUN25000CE"],
        "pe_symbols": ["NFO:NIFTY26JUN25000PE"],
        "symbols": ["NSE:NIFTY", "NFO:NIFTY26JUNFUT", "NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
        "all_symbols": ["NSE:NIFTY", "NFO:NIFTY26JUNFUT", "NFO:NIFTY26JUN25000CE", "NFO:NIFTY26JUN25000PE"],
        "all_tokens": [256265, 1001, 2001, 2002],
        "option_tokens": [2001, 2002],
        "token_by_symbol": {
            "NSE:NIFTY": 256265,
            "NFO:NIFTY26JUNFUT": 1001,
            "NFO:NIFTY26JUN25000CE": 2001,
            "NFO:NIFTY26JUN25000PE": 2002,
        },
        "symbol_by_token": {256265: "NSE:NIFTY", 1001: "NFO:NIFTY26JUNFUT", 2001: "NFO:NIFTY26JUN25000CE", 2002: "NFO:NIFTY26JUN25000PE"},
        "atm_strike": 25000,
    }


def _ctx(mdm: _Mdm) -> SimpleNamespace:
    runner = _Runner()
    return SimpleNamespace(
        active_trading_universe={},
        active_contract_basket=None,
        active_symbol_tokens={},
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        market_data_manager=mdm,
        data_hub=SimpleNamespace(set_active_contract_basket=lambda basket: None, subscribe_ticks=lambda *_a, **_k: None),
        strategy_runner=runner,
        strategy_manager=None,
        option_universe=None,
        instrument_manager=SimpleNamespace(is_loaded=lambda: True, get_token=lambda symbol: _basket()["token_by_symbol"][symbol]),
        broker_client=SimpleNamespace(get_instrument_token=lambda symbol: _basket()["token_by_symbol"].get(symbol, 256265)),
        settings=SimpleNamespace(option_universe=SimpleNamespace(strike_step=50), execution_mode="LIVE"),
        live_orders_armed=True,
        trading_ready=True,
        message_bus=SimpleNamespace(running=True),
        data_observation_ready=False,
    )


@pytest.mark.asyncio
async def test_spot_ws_timeout_uses_rest_fallback_for_basket_build(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    mdm = _Mdm(cached_ltp=25000.0)
    ctx = _ctx(mdm)
    monkeypatch.setattr(app, "_build_canonical_active_basket", lambda **_k: _basket())
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    spot = await app._wait_for_live_spot_or_raise(ctx, timeout=0.01, configured_mode="LIVE")
    await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=spot, configured_mode="LIVE", hydrate=True)

    assert ctx.active_contract_basket["selected_ce"] == "NFO:NIFTY26JUN25000CE"
    assert ctx.live_orders_armed is False
    assert ctx.live_block_reason is not None
    assert "STARTUP_SPOT_REST_FALLBACK_USED" in caplog.text


def test_rest_fallback_get_ltp_runtime_error_does_not_crash(caplog: pytest.LogCaptureFixture) -> None:
    class RuntimeMdm(_Mdm):
        def get_cached_ltp(self, *_a: Any, **_k: Any) -> float:
            return 0.0

        def get_ltp(self, *_a: Any, **_k: Any) -> float:
            raise RuntimeError("broker_unavailable")

        def refresh_quote_now(self, *_a: Any, **_k: Any) -> dict[str, float]:
            return {"last_price": 25000.0}

    caplog.set_level("WARNING", logger="nifty_scalper_bot.core.app")
    assert app._resolve_startup_rest_spot_ltp(_ctx(RuntimeMdm())) == 25000.0
    assert "STARTUP_SPOT_REST_FALLBACK_FAILED stage=get_ltp" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_spot_first_tick_routes_to_live_basket_build(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class TickMdm(_Mdm):
        def get_fresh_spot_tick(self, *_a: Any, **_k: Any) -> dict[str, Any]:
            return {"symbol": "NSE:NIFTY", "last_price": 25000.0, "source": "ws"}

    ctx = _ctx(TickMdm())
    monkeypatch.setattr(app, "_build_canonical_active_basket", lambda **_k: _basket())
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    await app._refresh_readiness_after_first_tick(ctx, reason="first_spot_tick_listener")

    assert ctx.active_contract_basket["selected_pe"] == "NFO:NIFTY26JUN25000PE"
    assert "LIVE_BASKET_BUILD_STARTED" in caplog.text
    assert "ACTIVE_CONTRACT_BASKET_COMMITTED" in caplog.text


@pytest.mark.asyncio
async def test_instrument_manager_not_ready_defers_then_retries(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(_Mdm(cached_ltp=25000.0))
    loaded = {"value": False}
    ctx.instrument_manager = SimpleNamespace(is_loaded=lambda: loaded["value"], get_token=lambda symbol: _basket()["token_by_symbol"][symbol])
    scheduled: list[Any] = []
    monkeypatch.setattr(app, "_build_canonical_active_basket", lambda **_k: _basket())
    monkeypatch.setattr(app, "_ensure_strategy_runner_started", lambda *_a, **_k: asyncio.sleep(0))
    monkeypatch.setattr(app, "_create_named_task", lambda coro, *, name: scheduled.append(coro) or SimpleNamespace(add_done_callback=lambda *_a: None))
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    first = await app._build_and_hydrate_live_basket_from_spot(ctx, spot_ltp=25000.0, configured_mode="LIVE")
    assert first["deferred"] is True
    loaded["value"] = True
    await app._deferred_basket_hydration_retry(ctx, configured_mode="LIVE", max_attempts=1, delay_seconds=0)

    assert ctx.active_contract_basket["selected_ce"] == "NFO:NIFTY26JUN25000CE"
    assert "LIVE_BASKET_BUILD_DEFERRED" in caplog.text
    assert "DEFERRED_BASKET_RETRY_FAILED attempt=1/1 reason=data_not_ready" in caplog.text
    assert "DEFERRED_BASKET_RETRY_SUCCESS" not in caplog.text
    for coro in scheduled:
        coro.close()


def test_selected_option_tokens_subscribed_after_basket_commit(caplog: pytest.LogCaptureFixture) -> None:
    mdm = _Mdm()
    ctx = _ctx(mdm)
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    app._commit_active_dynamic_basket(ctx, basket=_basket(), option_symbols=_basket()["option_symbols"], symbols=_basket()["symbols"], atm_strike=25000)

    assert {2001, 2002}.issubset(mdm.desired)
    assert "ACTIVE_BASKET_SUBSCRIPTION_RECONCILED" in caplog.text


def test_selected_option_token_resolution_accepts_bare_token_map(caplog: pytest.LogCaptureFixture) -> None:
    basket = _basket()
    basket["token_by_symbol"] = {
        "NIFTY26JUN25000CE": 2001,
        "NIFTY26JUN25000PE": 2002,
        "NSE:NIFTY": 256265,
        "NFO:NIFTY26JUNFUT": 1001,
    }
    basket["all_tokens"] = []
    mdm = _Mdm()
    ctx = _ctx(mdm)
    ctx.instrument_manager = None
    ctx.broker_client = None
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    app._commit_active_dynamic_basket(ctx, basket=basket, option_symbols=basket["option_symbols"], symbols=basket["symbols"], atm_strike=25000)

    assert ctx.live_block_reason != "option_token_missing:selected_ce"
    assert ctx.live_block_reason != "option_token_missing:selected_pe"
    assert {2001, 2002}.issubset(mdm.desired)
    assert "selected_ce_token=2001" in caplog.text
    assert "selected_pe_token=2002" in caplog.text


def test_selected_option_token_resolution_uses_explicit_selected_tokens(caplog: pytest.LogCaptureFixture) -> None:
    basket = _basket()
    basket["token_by_symbol"] = {"NSE:NIFTY": 256265, "NFO:NIFTY26JUNFUT": 1001}
    basket["all_tokens"] = [256265, 1001]
    basket["option_tokens"] = []
    basket["selected_ce_token"] = 2001
    basket["selected_pe_token"] = 2002
    mdm = _Mdm()
    ctx = _ctx(mdm)
    ctx.instrument_manager = None
    ctx.broker_client = None
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    app._commit_active_dynamic_basket(ctx, basket=basket, option_symbols=basket["option_symbols"], symbols=basket["symbols"], atm_strike=25000)

    assert ctx.live_block_reason != "option_token_missing:selected_ce"
    assert ctx.live_block_reason != "option_token_missing:selected_pe"
    assert "selected_ce_token=2001" in caplog.text
    assert "selected_pe_token=2002" in caplog.text


def test_hydration_runs_after_basket_commit() -> None:
    mdm = _Mdm(hydration={"hard_ready": False, "missing": ["NFO:NIFTY26JUN25000CE:quote_missing"], "symbols": {}})
    ctx = _ctx(mdm)

    app._commit_active_dynamic_basket(ctx, basket=_basket(), option_symbols=_basket()["option_symbols"], symbols=_basket()["symbols"], atm_strike=25000)

    call_names = [name for name, _ in mdm.calls]
    assert call_names.index("set_active_contract_basket") < call_names.index("hydrate_active_contract_basket")
    assert ctx.active_basket_hydration["hard_ready"] is False
    assert ctx.live_orders_armed is False


@pytest.mark.asyncio
async def test_live_readiness_receives_selected_option_context(monkeypatch: pytest.MonkeyPatch) -> None:
    mdm = _Mdm(hydration={"hard_ready": True, "missing": [], "symbols": {}})
    ctx = _ctx(mdm)
    app._commit_active_dynamic_basket(ctx, basket=_basket(), option_symbols=_basket()["option_symbols"], symbols=_basket()["symbols"], atm_strike=25000)
    monkeypatch.setattr(app, "_ensure_selected_options_hydrated", lambda *_a, **_k: asyncio.sleep(0))
    monkeypatch.setattr(app, "_runner_is_running", lambda _runner: True)

    await app._recompute_and_push_runtime_readiness(ctx, reason="active_basket_hydrated")

    assert ctx.strategy_runner.readiness["selected_ce"] == "NFO:NIFTY26JUN25000CE"
    assert ctx.strategy_runner.readiness["selected_pe"] == "NFO:NIFTY26JUN25000PE"
    assert "NFO:NIFTY26JUN25000CE" in ctx.strategy_runner.readiness["option_symbols"]


@pytest.mark.asyncio
async def test_no_silent_wait_after_startup_spot_token_request(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    ctx = _ctx(_Mdm(cached_ltp=0.0))
    tasks: list[Any] = []
    monkeypatch.setattr(app, "_create_named_task", lambda coro, *, name: tasks.append((name, coro)) or SimpleNamespace(add_done_callback=lambda *_a: None))
    caplog.set_level("INFO", logger="nifty_scalper_bot.core.app")

    with pytest.raises(RuntimeError, match="spot_ltp_unavailable"):
        await app._wait_for_live_spot_or_raise(ctx, timeout=0.01, configured_mode="LIVE")
    app._schedule_deferred_basket_retry(ctx, configured_mode="LIVE", reason="spot_ltp_unavailable")

    assert ctx.live_block_reason == "spot_ltp_unavailable"
    assert tasks and tasks[0][0] == "deferred_basket_hydration_retry"
    assert "STARTUP_SPOT_PROOF_TIMEOUT" in caplog.text
    # prevent un-awaited coroutine warnings from the captured scheduled retry
    tasks[0][1].close()
