from __future__ import annotations

from hypothesis import given, settings, strategies as st

from nifty_scalper_bot.execution.position_snapshot import (
    PositionSnapshotError,
    decode_position_snapshot,
)
from nifty_scalper_bot.risk.position_sizing import PositionSizer


@settings(max_examples=50, deadline=None)
@given(
    equity=st.floats(
        min_value=1_000.0,
        max_value=1_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    risk_per_trade_pct=st.floats(
        min_value=0.001,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    entry=st.floats(
        min_value=10.0,
        max_value=2_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    gap=st.floats(
        min_value=0.1,
        max_value=9.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    lot_size=st.sampled_from([25, 50, 65, 75]),
    confidence=st.floats(
        min_value=0.0,
        max_value=1.5,
        allow_nan=False,
        allow_infinity=False,
    ),
    regime_multiplier=st.floats(
        min_value=0.5,
        max_value=2.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    max_lots=st.integers(min_value=1, max_value=10),
    available_margin=st.floats(
        min_value=0.0,
        max_value=1_000_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    margin_per_lot=st.floats(
        min_value=1.0,
        max_value=100_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_position_sizer_preserves_lot_and_margin_invariants(
    equity: float,
    risk_per_trade_pct: float,
    entry: float,
    gap: float,
    lot_size: int,
    confidence: float,
    regime_multiplier: float,
    max_lots: int,
    available_margin: float,
    margin_per_lot: float,
) -> None:
    stop_loss = max(0.01, entry - gap)
    result = PositionSizer().size(
        equity=equity,
        risk_per_trade_pct=risk_per_trade_pct,
        entry=entry,
        stop_loss=stop_loss,
        lot_size=lot_size,
        confidence=confidence,
        regime_multiplier=regime_multiplier,
        max_lots=max_lots,
        available_margin=available_margin,
        margin_per_lot=margin_per_lot,
    )

    assert result.qty >= 0
    if not result.allowed:
        assert result.qty == 0
        return

    assert result.qty >= lot_size
    assert result.qty % lot_size == 0
    assert result.qty <= max_lots * lot_size
    affordable_lots = int(available_margin // max(margin_per_lot, 1.0))
    assert result.qty <= affordable_lots * lot_size


@settings(max_examples=50, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=999_999),
            st.integers(min_value=-10_000, max_value=10_000),
        ),
        min_size=0,
        max_size=12,
        unique_by=lambda item: item[0],
    )
)
def test_position_snapshot_round_trips_integral_broker_quantities(
    rows: list[tuple[int, int]],
) -> None:
    payload = {
        "net": [
            {
                "tradingsymbol": f"NIFTY{token}CE",
                "quantity": quantity,
                "product": "MIS",
            }
            for token, quantity in rows
        ]
    }

    snapshot = decode_position_snapshot(payload)

    assert len(snapshot.rows) == len(rows)
    for token, quantity in rows:
        symbol = f"NIFTY{token}CE"
        assert snapshot.quantity_for(symbol) == quantity
    assert snapshot.all_flat is all(quantity == 0 for _, quantity in rows)


@settings(max_examples=30, deadline=None)
@given(quantity=st.integers(min_value=-10_000, max_value=10_000))
def test_position_snapshot_rejects_duplicate_authority_rows(quantity: int) -> None:
    payload = {
        "net": [
            {"tradingsymbol": "NIFTY2692424500CE", "quantity": quantity},
            {"tradingsymbol": "NIFTY2692424500CE", "quantity": quantity},
        ]
    }

    try:
        decode_position_snapshot(payload)
    except PositionSnapshotError as exc:
        assert "duplicate broker position row" in str(exc)
    else:
        raise AssertionError("duplicate broker position rows must fail closed")
