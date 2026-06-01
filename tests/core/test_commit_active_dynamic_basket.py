from types import SimpleNamespace

from nifty_scalper_bot.core import app


def test_commit_active_dynamic_basket_selects_ce_pe_without_atm() -> None:
    ctx = SimpleNamespace(
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        active_trading_universe={},
        strategy_runner=None,
    )
    options = [
        'NFO:NIFTY2460019900CE',
        'NFO:NIFTY2460020000CE',
        'NFO:NIFTY2460019900PE',
        'NFO:NIFTY2460020000PE',
    ]
    selected_ce, selected_pe = app._commit_active_dynamic_basket(
        ctx,
        basket={},
        option_symbols=options,
        symbols=['NSE:NIFTY', *options],
        atm_strike=None,
    )
    assert selected_ce is not None and selected_ce.endswith('CE')
    assert selected_pe is not None and selected_pe.endswith('PE')


def test_commit_active_dynamic_basket_preserves_old_selected_when_valid() -> None:
    old_ce = 'NFO:NIFTY2460020000CE'
    old_pe = 'NFO:NIFTY2460020000PE'
    ctx = SimpleNamespace(
        selected_ce=old_ce,
        selected_pe=old_pe,
        atm_ce_symbol=old_ce,
        atm_pe_symbol=old_pe,
        active_trading_universe={},
        strategy_runner=None,
    )
    selected_ce, selected_pe = app._commit_active_dynamic_basket(
        ctx,
        basket={},
        option_symbols=[old_ce, old_pe],
        symbols=['NSE:NIFTY', old_ce, old_pe],
        atm_strike=None,
    )
    assert selected_ce == old_ce
    assert selected_pe == old_pe


def test_commit_active_dynamic_basket_preserves_requested_ssot_future_and_tokens() -> None:
    class _Mdm:
        def __init__(self) -> None:
            self.rotate_args = None
            self.received_basket = None

        def get_active_nifty_future_symbol_cached(self) -> str:
            raise AssertionError("MDM fallback must not override requested SSOT future")
        def purge_stale_nifty_futures(self, active_symbol, *, reason):
            return []
        def set_active_contract_basket(self, basket):
            self.received_basket = basket
        def maybe_rotate_nifty_futures_context_result(self, current_symbol, **kwargs):
            self.rotate_args = (current_symbol, kwargs)
            return None
    mdm = _Mdm()

    ctx = SimpleNamespace(
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        active_trading_universe={},
        strategy_runner=None,
        market_data_manager=mdm,
        strategy_manager=None,
    )
    ce = "NFO:NIFTY26JUN23900CE"
    pe = "NFO:NIFTY26JUN23900PE"
    app._commit_active_dynamic_basket(
        ctx,
        basket={
            "futures_symbol": "NFO:NIFTY26MAYFUT",
            "futures_token": 222,
            "all_symbols": ["NSE:NIFTY", "NFO:NIFTY26MAYFUT", ce, pe],
            "all_tokens": [256265, 222, 11, 12],
            "token_by_symbol": {"NSE:NIFTY": 256265, "NFO:NIFTY26MAYFUT": 222, ce: 11, pe: 12},
        },
        option_symbols=[ce, pe],
        symbols=["NSE:NIFTY", "NFO:NIFTY26MAYFUT", ce, pe],
        atm_strike=23900,
    )
    committed = ctx.active_trading_universe
    assert committed["futures_symbol"] == "NFO:NIFTY26MAYFUT"
    assert committed["futures_token"] == 222
    assert "NFO:NIFTY26MAYFUT" in committed["all_symbols"]
    assert 222 in committed["all_tokens"]
    assert committed["token_by_symbol"]["NFO:NIFTY26MAYFUT"] == 222
    assert mdm.received_basket is committed


def test_app_active_basket_uses_requested_future_not_mdm_fallback() -> None:
    class _Mdm:
        def __init__(self) -> None:
            self.purged: list[tuple[str, str]] = []
        def get_active_nifty_future_symbol_cached(self) -> str:
            return "NFO:NIFTY26JUNFUT"
        def purge_stale_nifty_futures(self, active_symbol, *, reason):
            self.purged.append((active_symbol, reason))
            return [111]
        def maybe_rotate_nifty_futures_context_result(self, current_symbol, **kwargs):
            self.rotated = (current_symbol, kwargs)
            return None

    mdm = _Mdm()
    ctx = SimpleNamespace(
        selected_ce=None,
        selected_pe=None,
        atm_ce_symbol=None,
        atm_pe_symbol=None,
        active_trading_universe={},
        strategy_runner=None,
        market_data_manager=mdm,
        strategy_manager=None,
    )
    ce = "NFO:NIFTY26JUN23900CE"
    pe = "NFO:NIFTY26JUN23900PE"
    app._commit_active_dynamic_basket(
        ctx,
        basket={"futures_symbol": "NFO:NIFTY26MAYFUT", "spot_symbol": "NSE:NIFTY"},
        option_symbols=[ce, pe],
        symbols=["NSE:NIFTY", "NFO:NIFTY26MAYFUT", ce, pe],
        atm_strike=23900,
    )

    committed = ctx.active_trading_universe
    assert committed["futures_symbol"] == "NFO:NIFTY26MAYFUT"
    assert "NFO:NIFTY26MAYFUT" in committed["symbols"]
    assert mdm.purged == [("NFO:NIFTY26MAYFUT", "active_dynamic_basket_commit")]
    assert mdm.rotated[0] == "NFO:NIFTY26MAYFUT"


def test_resolve_active_futures_for_basket_no_calendar_fallback(monkeypatch) -> None:
    class _Mdm:
        def get_active_nifty_future_symbol_cached(self):
            return None
        def resolve_active_nifty_future_symbol(self):
            return None

    monkeypatch.setattr(app, "_get_current_nifty_futures_symbol", lambda: (_ for _ in ()).throw(AssertionError("calendar fallback used")))
    ctx = SimpleNamespace(market_data_manager=_Mdm())

    assert app._resolve_active_futures_for_basket(ctx, "NFO:NIFTY26MAYFUT") == "NFO:NIFTY26MAYFUT"


def test_resolve_active_futures_for_basket_uses_active_universe_when_mdm_unresolved(monkeypatch) -> None:
    class _Mdm:
        def get_active_nifty_future_symbol_cached(self):
            return None
        def resolve_active_nifty_future_symbol(self):
            return None

    monkeypatch.setattr(app, "_get_current_nifty_futures_symbol", lambda: (_ for _ in ()).throw(AssertionError("calendar fallback used")))
    ctx = SimpleNamespace(
        market_data_manager=_Mdm(),
        active_trading_universe={"futures_symbol": "NFO:NIFTY26JUNFUT"},
        strategy_runner=None,
        strategy_manager=None,
    )

    assert app._resolve_active_futures_for_basket(ctx, "NFO:NIFTY26MAYFUT") == "NFO:NIFTY26MAYFUT"


def test_real_botcontext_active_symbol_tokens_and_selected_subscription_aliases() -> None:
    from nifty_scalper_bot.config.settings import Settings
    from nifty_scalper_bot.core.app import AppConfig, BotContext, RateLimiter

    class _Mdm:
        def __init__(self) -> None:
            self.desired: set[int] = set()
            self.received_basket = None

        def purge_stale_nifty_futures(self, active_symbol, *, reason):
            return []

        def set_active_contract_basket(self, basket):
            self.received_basket = basket
            self.desired.update(int(t) for t in basket.get("all_tokens", []) if t)

        def request_token_subscription(self, token: int, symbol: str | None = None) -> bool:
            self.desired.add(int(token))
            return True

        def desired_token_count(self) -> int:
            return len(self.desired)

        def desired_tokens_snapshot(self) -> list[int]:
            return sorted(self.desired)

        def ws_token_count(self):
            return None

    mdm = _Mdm()
    ctx = BotContext(
        settings=Settings(),
        config=AppConfig(),
        rate_limiter=RateLimiter(),
        broker_client=None,
        websocket_client=None,
        websocket_manager=None,
        streamer=None,
        stream_supervisor=None,
        polling_fallback_streamer=None,
        message_bus=None,
        market_data_manager=mdm,
        strategy_runner=None,
        strategy_manager=None,
    )
    assert hasattr(ctx, "active_symbol_tokens")

    ce = "NFO:NIFTY26JUN23900CE"
    pe = "NFO:NIFTY26JUN23900PE"
    app._commit_active_dynamic_basket(
        ctx,
        basket={
            "spot_symbol": "NSE:NIFTY",
            "spot_token": 256265,
            "selected_ce": ce,
            "selected_pe": pe,
            "selected_ce_token": 11,
            "selected_pe_token": 12,
            "all_symbols": ["NSE:NIFTY", ce, pe],
            "all_tokens": [256265, 11, 12],
            "token_by_symbol": {"NSE:NIFTY": 256265, ce: 11, pe: 12},
        },
        option_symbols=[ce, pe],
        symbols=["NSE:NIFTY", ce, pe],
        atm_strike=23900,
    )

    assert ctx.active_symbol_tokens[ce] == 11
    assert ctx.active_symbol_tokens["NIFTY26JUN23900CE"] == 11
    assert ctx.active_symbol_tokens[pe] == 12
    assert ctx.active_symbol_tokens["NIFTY26JUN23900PE"] == 12
    assert {11, 12}.issubset(set(mdm.desired_tokens_snapshot()))
