"""Tests for Zerodha margin payload normalization and extraction."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient


def _build_client() -> ZerodhaKiteClient:
    """Create a ZerodhaKiteClient instance without invoking __init__."""

    return ZerodhaKiteClient.__new__(ZerodhaKiteClient)


def test_resolve_margin_payload_handles_nested_segment() -> None:
    """Ensure nested margin payloads for segments are resolved correctly."""

    client = _build_client()
    payload: Mapping[str, Any] = {
        "equity": {
            "available": {
                "cash": 450_000.0,
                "live_balance": 450_000.0,
            },
            "utilised": {"debits": 50_000.0},
            "net": 500_000.0,
        },
        "commodity": {
            "available": {"cash": 10_000.0},
            "net": 10_000.0,
        },
    }

    resolved = client._resolve_margin_payload(payload, segment="equity")
    assert resolved["available"]["cash"] == 450_000.0
    assert resolved["net"] == 500_000.0


def test_resolve_margin_payload_prefers_data_mapping() -> None:
    """Ensure payloads using the data key are returned as-is."""

    client = _build_client()
    payload: Mapping[str, Any] = {
        "data": {
            "available": {"cash": 125_000.0},
            "net": 150_000.0,
        }
    }

    resolved = client._resolve_margin_payload(payload, segment="equity")
    assert resolved["available"]["cash"] == 125_000.0
    assert resolved["net"] == 150_000.0


def test_normalize_margin_payload_extracts_nested_available() -> None:
    """Normalize margin payload to summary with nested available fields."""

    client = _build_client()
    payload: Mapping[str, Any] = {
        "equity": {
            "available": {
                "cash": 200_000.0,
                "live_balance": 210_000.0,
            },
            "utilised": {"debits": 25_000.0},
            "net": 230_000.0,
        }
    }

    summary = client._normalize_margin_payload(payload, segment="equity")
    assert summary == {
        "available": 210_000.0,
        "used": 25_000.0,
        "net": 230_000.0,
    }


def test_normalize_margin_payload_supports_available_cash() -> None:
    """Ensure available_cash fields are treated as deployable balance."""

    client = _build_client()
    payload: Mapping[str, Any] = {
        "equity": {
            "available_cash": 175_000.0,
            "utilised": {"debits": 0.0},
        }
    }

    summary = client._normalize_margin_payload(payload, segment="equity")

    assert summary["available"] == pytest.approx(175_000.0)
    assert summary["net"] == pytest.approx(175_000.0)



def test_negative_utilised_debits_does_not_fail_refresh() -> None:
    """A negative utilised.debits (legit reversal/credit) must NOT fail the whole
    balance refresh — it previously raised negative_margin_field and cascaded to
    BROKER NOT READY."""
    client = _build_client()
    payload: Mapping[str, Any] = {
        "equity": {
            "available": {"cash": 16_248.60, "live_balance": 16_248.60},
            "utilised": {"debits": -1_500.0, "exposure": 500.0},
            "net": 16_248.60,
        }
    }
    summary = client._normalize_margin_payload(payload, segment="equity")
    assert summary["available"] == 16_248.60  # balance preserved
    assert summary["used"] == -1_000.0  # -1500 + 500, negative debits honoured


def test_negative_available_cash_still_rejected() -> None:
    """Available (deployable) balance must remain strictly non-negative."""
    client = _build_client()
    payload: Mapping[str, Any] = {
        "equity": {
            "available": {"cash": -100.0},
            "utilised": {"debits": 0.0},
        }
    }
    with pytest.raises(Exception):
        client._normalize_margin_payload(payload, segment="equity")
