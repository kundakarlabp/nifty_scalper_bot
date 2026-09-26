from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pytest

from nifty_scalper_bot.backtesting.replay import (
    HistoricalContractCatalog,
    ReplayContractSnapshot,
    ReplayHarness,
)

ROOT = Path(__file__).resolve().parents[2]
REPLAY_DIR = ROOT / "tests" / "fixtures" / "replay"
OPTION_SYMBOL = "NFO:NIFTY26O0124900CE"
INDEX_SYMBOL = "NSE:NIFTY 50"


class _Hub:
    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, Any]] = {}
        self.baskets: list[dict[str, Any]] = []

    def store_quote(
        self,
        symbol: str,
        quote: Mapping[str, Any],
        *,
        source: str,
    ) -> None:
        payload = dict(quote)
        payload["source"] = source
        self.quotes[symbol] = payload

    def set_active_contract_basket(self, basket: Mapping[str, Any]) -> None:
        self.baskets.append(dict(basket))


class _Paper:
    def __init__(self) -> None:
        self.hub = _Hub()
        self.processed: list[str] = []

    def process_quote(self, symbol: str) -> None:
        self.processed.append(symbol)

    def get_orders(self) -> list[dict[str, Any]]:
        return []


class _Runner:
    def __init__(self) -> None:
        self.ticks: list[dict[str, Any]] = []
        self.baskets: list[dict[str, Any]] = []

    def set_active_trading_universe(self, basket: Mapping[str, Any]) -> None:
        self.baskets.append(dict(basket))

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        self.ticks.append(dict(tick))

    def get_status(self) -> dict[str, Any]:
        return {"symbols": {}}


def _catalog() -> HistoricalContractCatalog:
    payload = json.loads(
        (REPLAY_DIR / "golden_contract_catalog.json").read_text(encoding="utf-8")
    )
    snapshots = [
        ReplayContractSnapshot(
            available_from=datetime.fromisoformat(item["available_from"]),
            basket=item["basket"],
        )
        for item in payload["snapshots"]
    ]
    return HistoricalContractCatalog(snapshots)


def test_golden_market_path_replays_through_existing_harness() -> None:
    runner = _Runner()
    paper = _Paper()
    harness = ReplayHarness(
        runner,
        paper,  # type: ignore[arg-type]
        option_symbol=OPTION_SYMBOL,
        index_symbol=INDEX_SYMBOL,
        contract_catalog=_catalog(),
    )

    result = harness.run_file(REPLAY_DIR / "golden_market_path.csv")

    assert result.bars_processed == 3
    assert result.orders == []
    assert len(runner.ticks) == 6
    assert [tick["symbol"] for tick in runner.ticks] == [
        INDEX_SYMBOL,
        OPTION_SYMBOL,
        INDEX_SYMBOL,
        OPTION_SYMBOL,
        INDEX_SYMBOL,
        OPTION_SYMBOL,
    ]
    option_ticks = [tick for tick in runner.ticks if tick["symbol"] == OPTION_SYMBOL]
    assert all(tick["source"] == "historical_replay" for tick in option_ticks)
    assert [tick["instrument_token"] for tick in option_ticks] == [111001] * 3
    assert all(tick["tradable_quote"] in (True, 1) for tick in option_ticks)
    assert all(float(tick["ask"]) > float(tick["bid"]) for tick in option_ticks)
    assert runner.baskets[-1]["basket_version"] == "golden-v1"
    assert paper.hub.baskets[-1]["selected_ce_token"] == 111001
    assert paper.hub.quotes[OPTION_SYMBOL]["source"] == "historical_replay"


def test_golden_contract_catalog_fails_closed_before_first_snapshot() -> None:
    catalog = _catalog()

    with pytest.raises(LookupError, match="no contract basket available"):
        catalog.resolve(datetime.fromisoformat("2026-09-24T09:14:59+05:30"))

    basket = catalog.resolve(datetime.fromisoformat("2026-09-24T09:15:00+05:30"))
    assert basket["selected_ce"] == OPTION_SYMBOL
    assert basket["selected_ce_token"] == 111001
