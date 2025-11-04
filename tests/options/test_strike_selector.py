from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from nifty_scalper_bot.config.settings import LiquiditySettings, SelectorSettings
from nifty_scalper_bot.options.strike_selector import StrikeSelector


class _StubHub:
    def __init__(
        self,
        chain: list[dict[str, Any]],
        *,
        mdm: Any | None = None,
        resolver: Any | None = None,
    ) -> None:
        self._chain = chain
        self._mdm = mdm
        self.resolver = resolver
        self.requests: list[str] = []

    def get_option_chain(
        self, expiry: str, *, force_refresh: bool = False
    ) -> list[dict[str, Any]]:
        self.requests.append(expiry)
        return list(self._chain)


class _StubResolver:
    def __init__(self, payload: dict[str, dict[str, Any]]) -> None:
        self._payload = payload

    def lookup(self, symbol: str) -> dict[str, Any] | None:
        return self._payload.get(symbol)


class _StubMDM:
    def __init__(self, quotes: dict[str, dict[str, Any]]) -> None:
        self._quotes = quotes
        self.refresh_requests: list[str] = []

    def get_quote(self, symbol: str) -> dict[str, Any] | None:
        return self._quotes.get(symbol)

    def refresh_quote_now(self, symbol: str) -> dict[str, Any] | None:
        self.refresh_requests.append(symbol)
        return self._quotes.get(symbol)


def _base_chain() -> list[dict[str, Any]]:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    return [
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 152.5,
            "delta": 0.52,
            "bid": 151.5,
            "ask": 153.0,
            "open_interest": 75000,
            "trades": 120,
            "depth": {
                "buy": [{"quantity": 25}, {"quantity": 30}, {"quantity": 20}],
                "sell": [{"quantity": 22}, {"quantity": 18}, {"quantity": 16}],
            },
        },
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.45,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 82000,
            "trades": 150,
            "depth": {
                "buy": [{"quantity": 40}, {"quantity": 35}, {"quantity": 25}],
                "sell": [{"quantity": 32}, {"quantity": 24}, {"quantity": 20}],
            },
        },
    ]


@pytest.mark.parametrize(
    ("side", "option_type"),
    [
        ("BUY", "CE"),
        ("BUY", "PE"),
        ("SELL", "CE"),
        ("SELL", "PE"),
    ],
)
def test_select_contract_supports_all_combinations(side: str, option_type: str) -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = _base_chain() + [
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 118.0,
            "delta": -0.34,
            "bid": 117.5,
            "ask": 118.5,
            "open_interest": 76000,
            "trades": 140,
            "depth": {
                "buy": [{"quantity": 38}, {"quantity": 34}, {"quantity": 28}],
                "sell": [{"quantity": 30}, {"quantity": 26}, {"quantity": 22}],
            },
        }
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side=side,
        option_type=option_type,
        underlying_price=19850.0,
    )

    assert selected is not None
    assert selected.option_type == option_type


def test_selects_atm_contract_for_buy() -> None:
    hub = _StubHub(_base_chain())
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )

    assert selected is not None
    assert selected.symbol == "NFO:NIFTY24JUL19900CE"
    assert hub.requests == ["weekly"]


def test_selects_delta_target_for_sell() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 95.0,
            "delta": -0.28,
            "bid": 94.5,
            "ask": 95.5,
            "open_interest": 65000,
            "trades": 130,
            "depth": {
                "buy": [{"quantity": 40}, {"quantity": 35}, {"quantity": 25}],
                "sell": [{"quantity": 32}, {"quantity": 24}, {"quantity": 20}],
            },
        },
        {
            "tradingsymbol": "NIFTY24JUL19900PE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 135.0,
            "delta": -0.42,
            "bid": 134.0,
            "ask": 136.0,
            "open_interest": 72000,
            "trades": 110,
            "depth": {
                "buy": [{"quantity": 30}, {"quantity": 28}, {"quantity": 24}],
                "sell": [{"quantity": 26}, {"quantity": 22}, {"quantity": 18}],
            },
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="DELTA_TARGET", delta_target=0.3, delta_tolerance=0.05, expiry="weekly"
        ),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.5,
            min_trades=80,
            min_top3_depth=40,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="SELL",
        underlying_price=19840.0,
        option_type="PE",
    )

    assert selected is not None
    assert selected.symbol == "NFO:NIFTY24JUL19800PE"


def test_explicit_option_type_without_legacy_mapping() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.45,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 82000,
            "trades": 150,
            "depth": {
                "buy": [{"quantity": 40}, {"quantity": 35}, {"quantity": 25}],
                "sell": [{"quantity": 32}, {"quantity": 24}, {"quantity": 20}],
            },
        },
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 118.0,
            "delta": -0.34,
            "bid": 117.5,
            "ask": 118.5,
            "open_interest": 76000,
            "trades": 140,
            "depth": {
                "buy": [{"quantity": 38}, {"quantity": 34}, {"quantity": 28}],
                "sell": [{"quantity": 30}, {"quantity": 26}, {"quantity": 22}],
            },
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="ATM", expiry="weekly", legacy_side_to_type=False
        ),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19850.0,
        option_type="PE",
    )

    assert selected is not None
    assert selected.option_type == "PE"
    assert selected.symbol == "NFO:NIFTY24JUL19800PE"


def test_liquidity_filters_skip_weak_contracts() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 150.0,
            "delta": 0.51,
            "bid": 149.0,
            "ask": 151.0,
            "open_interest": 10000,
            "trades": 5,
            "depth": {"buy": [{"quantity": 5}], "sell": [{"quantity": 5}]},
        },
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.43,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 80000,
            "trades": 120,
            "depth": {
                "buy": [{"quantity": 40}, {"quantity": 35}, {"quantity": 25}],
                "sell": [{"quantity": 32}, {"quantity": 24}, {"quantity": 20}],
            },
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19820.0,
        option_type="CE",
    )

    assert selected is not None
    assert selected.symbol == "NFO:NIFTY24JUL19900CE"


def test_returns_none_when_filters_reject_all() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 150.0,
            "delta": 0.51,
            "bid": 148.0,
            "ask": 154.0,
            "open_interest": 10000,
            "trades": 5,
            "depth": {"buy": [{"quantity": 5}], "sell": [{"quantity": 5}]},
        },
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.43,
            "bid": 100.0,
            "ask": 120.0,
            "open_interest": 40000,
            "trades": 20,
            "depth": {"buy": [{"quantity": 10}], "sell": [{"quantity": 8}]},
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=1.5,
            min_trades=80,
            min_top3_depth=80,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19850.0,
        option_type="CE",
    )

    assert selected is None


def test_option_type_matches_trade_side() -> None:
    chain = _base_chain() + [
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": datetime(2024, 7, 18, tzinfo=timezone.utc).isoformat(),
            "ltp": 95.0,
            "delta": -0.32,
            "bid": 94.5,
            "ask": 95.5,
            "open_interest": 82000,
            "trades": 120,
            "depth": {
                "buy": [{"quantity": 25}, {"quantity": 20}, {"quantity": 18}],
                "sell": [{"quantity": 22}, {"quantity": 16}, {"quantity": 14}],
            },
        }
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.5,
            min_trades=50,
            min_top3_depth=40,
        ),
    )

    long_selection = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )
    short_selection = selector.select_contract(
        underlying="NIFTY",
        side="SELL",
        underlying_price=19820.0,
        option_type="PE",
    )

    assert long_selection is not None
    assert short_selection is not None
    assert long_selection.option_type == "CE"
    assert short_selection.option_type == "PE"


def test_blank_option_type_returns_none_without_legacy() -> None:
    hub = _StubHub(_base_chain())
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="ATM", expiry="weekly", legacy_side_to_type=False
        ),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        option_type=" ",
        underlying_price=19860.0,
    )

    assert selected is None


def test_legacy_option_mapping_not_applied() -> None:
    chain = _base_chain() + [
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": datetime(2024, 7, 18, tzinfo=timezone.utc).isoformat(),
            "ltp": 95.0,
            "delta": -0.32,
            "bid": 94.5,
            "ask": 95.5,
            "open_interest": 82000,
            "trades": 120,
            "depth": {
                "buy": [{"quantity": 25}, {"quantity": 20}, {"quantity": 18}],
                "sell": [{"quantity": 22}, {"quantity": 16}, {"quantity": 14}],
            },
        }
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="ATM", expiry="weekly", legacy_side_to_type=True
        ),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.5,
            min_trades=50,
            min_top3_depth=40,
        ),
    )

    selection = selector.select_contract(
        underlying="NIFTY",
        side="SELL",
        option_type="",
        underlying_price=19820.0,
    )

    assert selection is None


def test_selector_normalizes_symbol_and_enriches_metadata() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.45,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 82000,
            "trades": 150,
            "depth": {
                "buy": [
                    {"quantity": 40},
                    {"quantity": 35},
                    {"quantity": 25},
                ],
                "sell": [
                    {"quantity": 32},
                    {"quantity": 24},
                    {"quantity": 20},
                ],
            },
        }
    ]
    resolver = _StubResolver(
        {
            "NFO:NIFTY24JUL19900CE": {
                "tradingsymbol": "NIFTY24JUL19900CE",
                "exchange": "NFO",
                "instrument_token": 654321,
            }
        }
    )
    mdm = _StubMDM({"NFO:NIFTY24JUL19900CE": {"ltp": 110.0}})
    hub = _StubHub(chain, mdm=mdm, resolver=resolver)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )

    assert selected is not None
    assert selected.symbol == "NFO:NIFTY24JUL19900CE"
    assert selected.metadata.get("instrument_token") == 654321


def test_selector_returns_none_when_quotes_unavailable() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.45,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 82000,
            "trades": 150,
            "depth": {
                "buy": [
                    {"quantity": 40},
                    {"quantity": 35},
                    {"quantity": 25},
                ],
                "sell": [
                    {"quantity": 32},
                    {"quantity": 24},
                    {"quantity": 20},
                ],
            },
        }
    ]
    mdm = _StubMDM({})
    hub = _StubHub(chain, mdm=mdm)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )

    assert selected is None
    assert mdm.refresh_requests == ["NFO:NIFTY24JUL19900CE"]


def test_selector_skips_unquoteable_and_chooses_next_candidate() -> None:
    expiry = datetime(2024, 7, 18, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19900CE",
            "strike": 19900,
            "expiry": expiry.isoformat(),
            "ltp": 110.0,
            "delta": 0.45,
            "bid": 109.0,
            "ask": 111.0,
            "open_interest": 82000,
            "trades": 150,
            "depth": {
                "buy": [
                    {"quantity": 40},
                    {"quantity": 35},
                    {"quantity": 25},
                ],
                "sell": [
                    {"quantity": 32},
                    {"quantity": 24},
                    {"quantity": 20},
                ],
            },
        },
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": expiry.isoformat(),
            "ltp": 152.5,
            "delta": 0.52,
            "bid": 151.5,
            "ask": 153.0,
            "open_interest": 75000,
            "trades": 120,
            "depth": {
                "buy": [
                    {"quantity": 25},
                    {"quantity": 30},
                    {"quantity": 20},
                ],
                "sell": [
                    {"quantity": 22},
                    {"quantity": 18},
                    {"quantity": 16},
                ],
            },
        },
    ]
    mdm = _StubMDM({"NFO:NIFTY24JUL19800CE": {"ltp": 152.5}})
    hub = _StubHub(chain, mdm=mdm)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )

    assert selected is not None
    assert selected.symbol == "NFO:NIFTY24JUL19800CE"
    assert mdm.refresh_requests == ["NFO:NIFTY24JUL19900CE"]


def test_selector_parses_epoch_expiry_values() -> None:
    expiry_dt = datetime(2024, 7, 18, tzinfo=timezone.utc)
    expiry_epoch_ms = int(expiry_dt.timestamp() * 1000)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": expiry_epoch_ms,
            "ltp": 152.5,
            "delta": 0.52,
            "bid": 151.5,
            "ask": 153.0,
            "open_interest": 75000,
            "trades": 120,
            "depth": {
                "buy": [
                    {"quantity": 25},
                    {"quantity": 30},
                    {"quantity": 20},
                ],
                "sell": [
                    {"quantity": 22},
                    {"quantity": 18},
                    {"quantity": 16},
                ],
            },
        }
    ]
    mdm = _StubMDM({"NFO:NIFTY24JUL19800CE": {"ltp": 152.5}})
    hub = _StubHub(chain, mdm=mdm)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(mode="ATM", expiry="weekly"),
        liquidity_settings=LiquiditySettings(
            min_oi=50000,
            max_spread_pct=2.0,
            min_trades=50,
            min_top3_depth=60,
        ),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        underlying_price=19860.0,
        option_type="CE",
    )

    assert selected is not None
    assert selected.expiry == expiry_dt


def test_auto_expiry_regime_rolls_to_next() -> None:
    weekly_expiry = datetime(2024, 7, 18, 15, 0, tzinfo=timezone.utc)
    monthly_expiry = datetime(2024, 7, 25, 15, 0, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800CE",
            "strike": 19800,
            "expiry": weekly_expiry.isoformat(),
            "ltp": 120.0,
            "delta": 0.5,
            "bid": 119.5,
            "ask": 120.5,
            "open_interest": 60000,
            "trades": 80,
            "depth": {"buy": [{"quantity": 30}], "sell": [{"quantity": 28}]},
        },
        {
            "tradingsymbol": "NIFTY24JUL19800CE-MONTHLY",
            "strike": 19800,
            "expiry": monthly_expiry.isoformat(),
            "ltp": 180.0,
            "delta": 0.52,
            "bid": 179.0,
            "ask": 181.0,
            "open_interest": 85000,
            "trades": 120,
            "depth": {"buy": [{"quantity": 40}], "sell": [{"quantity": 35}]},
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="ATM",
            expiry="weekly",
            expiry_regime="auto",
            expiry_roll_minutes=60,
        ),
        liquidity_settings=LiquiditySettings(),
        clock=lambda: weekly_expiry - timedelta(minutes=45),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="BUY",
        option_type="CE",
        underlying_price=19850.0,
    )

    assert selected is not None
    assert selected.expiry.date() == monthly_expiry.date()


def test_special_expiry_regime_prefers_configured_date() -> None:
    weekly_expiry = datetime(2024, 7, 18, 15, 0, tzinfo=timezone.utc)
    special_expiry = datetime(2024, 7, 19, 15, 0, tzinfo=timezone.utc)
    chain = [
        {
            "tradingsymbol": "NIFTY24JUL19800PE",
            "strike": 19800,
            "expiry": weekly_expiry.isoformat(),
            "ltp": 90.0,
            "delta": -0.35,
            "bid": 89.5,
            "ask": 90.5,
            "open_interest": 62000,
            "trades": 70,
            "depth": {"buy": [{"quantity": 25}], "sell": [{"quantity": 21}]},
        },
        {
            "tradingsymbol": "NIFTY24JUL19800PE-SPECIAL",
            "strike": 19800,
            "expiry": special_expiry.isoformat(),
            "ltp": 95.0,
            "delta": -0.32,
            "bid": 94.0,
            "ask": 96.0,
            "open_interest": 90000,
            "trades": 95,
            "depth": {"buy": [{"quantity": 35}], "sell": [{"quantity": 30}]},
        },
    ]
    hub = _StubHub(chain)
    selector = StrikeSelector(
        data_hub=hub,
        selector_settings=SelectorSettings(
            mode="ATM",
            expiry="weekly",
            expiry_regime="special",
            special_expiries=(special_expiry.date().isoformat(),),
        ),
        liquidity_settings=LiquiditySettings(),
        clock=lambda: weekly_expiry - timedelta(hours=2),
    )

    selected = selector.select_contract(
        underlying="NIFTY",
        side="SELL",
        option_type="PE",
        underlying_price=19800.0,
    )

    assert selected is not None
    assert selected.expiry.date() == special_expiry.date()
