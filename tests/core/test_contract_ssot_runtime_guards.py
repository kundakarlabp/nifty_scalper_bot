from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.core import app
from nifty_scalper_bot.core.contract_selector import get_atm_contracts
from nifty_scalper_bot.data.universe_builder import build_nifty_universe
from nifty_scalper_bot.instruments.active_contracts import derive_nifty_future_from_selected_options
from nifty_scalper_bot.options.resolver import OptionResolver


class Broker:
    def instruments(self, exchange):
        raise AssertionError(f"broker instruments must not be called in live path: {exchange}")


def test_live_deprecated_resolvers_do_not_become_authority(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    with pytest.raises(RuntimeError, match="requires InstrumentManager"):
        OptionResolver(Broker()).resolve_atm_pair(25000)
    with pytest.raises(RuntimeError, match="fallback disabled"):
        get_atm_contracts(Broker(), 25000)
    with pytest.raises(RuntimeError, match="InstrumentManager ActiveContractBasket"):
        build_nifty_universe(Broker(), 25000)
    with pytest.raises(RuntimeError, match="diagnostic only"):
        derive_nifty_future_from_selected_options(["NFO:NIFTY26JUN25000CE"])
    with pytest.raises(RuntimeError, match="calendar futures generation disabled"):
        app._get_current_nifty_futures_symbol()


def test_strategy_sources_do_not_call_broker_instruments_or_historical_data_directly():
    strategy_files = [
        Path("src/nifty_scalper_bot/core/strategy_manager.py"),
        Path("src/nifty_scalper_bot/strategies/signal_generator.py"),
        Path("src/nifty_scalper_bot/strategies/runner.py"),
    ]
    for path in strategy_files:
        text = path.read_text()
        assert '.instruments("NFO")' not in text
        assert ".historical_data(" not in text
        assert "OptionResolver(" not in text
        assert "InstrumentsCache(" not in text


def test_no_runtime_calendar_future_generation_outside_guarded_helpers():
    app_text = Path("src/nifty_scalper_bot/core/app.py").read_text()
    assert 'return f"NFO:NIFTY' not in app_text
    assert "calendar futures generation disabled in LIVE" in app_text


def test_readme_architecture_section_not_duplicated():
    text = Path("README.md").read_text()
    assert text.count("## Runtime Contract/Data Architecture") == 1
    assert text.count("## Runtime Troubleshooting") == 1


def test_resolve_active_future_accepts_requested_before_fallback(monkeypatch):
    class Mdm:
        def get_active_nifty_future_symbol_cached(self):
            raise AssertionError("fallback should not be called when requested future is canonical")
    ctx = SimpleNamespace(market_data_manager=Mdm())
    assert app._resolve_active_futures_for_basket(ctx, "NFO:NIFTY26JUNFUT") == "NFO:NIFTY26JUNFUT"
