from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import _merge_orderflow_quote_context


def test_orderflow_quote_context_copies_only_canonical_ofi_fields() -> None:
    indicators: dict[str, object] = {"direction_bias": "CE"}
    _merge_orderflow_quote_context(
        indicators,
        {
            "ofi_ready": True,
            "ofi_event": 20.0,
            "ofi_1s": 60.0,
            "ofi_1s_normalized": 0.30,
            "ofi_update_count_1s": 4,
            "ofi_source": "ws_full_depth",
            "queue_imbalance_top": 0.15,
            "unrelated_quote_field": "ignored",
        },
    )

    assert indicators["direction_bias"] == "CE"
    assert indicators["ofi_ready"] is True
    assert indicators["ofi_1s_normalized"] == 0.30
    assert indicators["ofi_update_count_1s"] == 4
    assert indicators["ofi_source"] == "ws_full_depth"
    assert "unrelated_quote_field" not in indicators


def test_orderflow_quote_context_does_not_overwrite_with_missing_values() -> None:
    indicators: dict[str, object] = {"ofi_1s_normalized": 0.25}

    _merge_orderflow_quote_context(
        indicators,
        {"ofi_1s_normalized": None, "ofi_ready": False},
    )

    assert indicators["ofi_1s_normalized"] == 0.25
    assert indicators["ofi_ready"] is False
