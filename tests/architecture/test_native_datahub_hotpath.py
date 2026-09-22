from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.core.runtime_reliability_hardening import (
    _is_canonical_runtime_tick,
)
from nifty_scalper_bot.data.data_hub import DataHub


def test_datahub_hotpath_is_native_not_runtime_monkey_patched() -> None:
    reliability = Path(
        "src/nifty_scalper_bot/core/runtime_reliability_hardening.py"
    ).read_text(encoding="utf-8")
    datahub = Path("src/nifty_scalper_bot/data/data_hub.py").read_text(
        encoding="utf-8"
    )

    assert "DataHub._canonicalize_tick_payload =" not in reliability
    assert "if _is_canonical_runtime_tick(payload):" in datahub
    assert callable(DataHub._canonicalize_tick_payload)


def test_canonical_runtime_tick_contract_remains_available() -> None:
    tick = {
        "symbol": "NFO:NIFTY26SEP23350CE",
        "instrument_token": 123,
        "ltp": 100.0,
        "timestamp": "2026-09-22T09:45:00+00:00",
        "timestamp_ms": 1790070300000.0,
        "timestamp_source": "exchange_timestamp",
        "timestamp_quality": "exchange",
        "source_timestamp_valid": True,
        "received_at": 1790070300.0,
        "source": "ws_full",
        "depth_available": True,
        "tradable_quote": True,
    }

    assert _is_canonical_runtime_tick(tick) is True
