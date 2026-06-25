from __future__ import annotations

import math

import pytest

from nifty_scalper_bot.execution.fill_ledger import (
    BracketFillLedgerStore,
    FillConflictError,
    FillLeg,
    FillValidationError,
    calculate_realized_pnl,
)


def _leg(
    *,
    fill_id: str,
    order_id: str,
    kind: str,
    side: str,
    quantity: int,
    price: float,
    fees: float = 0.0,
    target: str | None = None,
    recorded_at: float = 1_700_000_000.0,
) -> FillLeg:
    return FillLeg(
        fill_id=fill_id,
        bracket_id="entry-1",
        order_id=order_id,
        kind=kind,  # type: ignore[arg-type]
        side=side,  # type: ignore[arg-type]
        quantity=quantity,
        price=price,
        fees=fees,
        target=target,
        recorded_at=recorded_at,
    )


def test_rejects_invalid_quantity_and_price() -> None:
    with pytest.raises(FillValidationError):
        _leg(
            fill_id="bad-q",
            order_id="entry",
            kind="ENTRY",
            side="BUY",
            quantity=0,
            price=100.0,
        )
    with pytest.raises(FillValidationError):
        _leg(
            fill_id="bad-p",
            order_id="entry",
            kind="ENTRY",
            side="BUY",
            quantity=65,
            price=float("nan"),
        )


def test_replay_is_idempotent_but_conflict_fails(tmp_path) -> None:
    store = BracketFillLedgerStore(tmp_path / "fills.db")
    leg = _leg(
        fill_id="trade-1",
        order_id="entry-order",
        kind="ENTRY",
        side="BUY",
        quantity=65,
        price=100.0,
    )
    assert store.record_fill(leg) is True
    assert store.record_fill(leg) is False
    with pytest.raises(FillConflictError):
        store.record_fill(
            _leg(
                fill_id="trade-1",
                order_id="entry-order",
                kind="ENTRY",
                side="BUY",
                quantity=65,
                price=101.0,
            )
        )


def test_scaled_long_exit_uses_each_confirmed_fill() -> None:
    pnl = calculate_realized_pnl(
        [
            _leg(
                fill_id="entry",
                order_id="entry-order",
                kind="ENTRY",
                side="BUY",
                quantity=65,
                price=100.0,
                fees=3.0,
            ),
            _leg(
                fill_id="tp1",
                order_id="tp1-order",
                kind="EXIT",
                side="SELL",
                quantity=25,
                price=110.0,
                fees=2.0,
                target="TP1",
            ),
            _leg(
                fill_id="final",
                order_id="final-order",
                kind="EXIT",
                side="SELL",
                quantity=40,
                price=95.0,
                fees=2.0,
                target="SL",
            ),
        ]
    )
    assert pnl.entry_quantity == 65
    assert pnl.exit_quantity == 65
    assert pnl.entry_vwap == 100.0
    assert math.isclose(pnl.exit_vwap or 0.0, 6550.0 / 65.0, rel_tol=1e-8)
    assert pnl.gross_pnl == 50.0
    assert pnl.fees == 7.0
    assert pnl.net_pnl == 43.0
    assert pnl.complete is True


def test_partial_exit_is_not_complete() -> None:
    pnl = calculate_realized_pnl(
        [
            _leg(
                fill_id="entry",
                order_id="entry-order",
                kind="ENTRY",
                side="BUY",
                quantity=65,
                price=100.0,
            ),
            _leg(
                fill_id="tp1",
                order_id="tp1-order",
                kind="EXIT",
                side="SELL",
                quantity=25,
                price=110.0,
            ),
        ]
    )
    assert pnl.gross_pnl == 250.0
    assert pnl.exit_quantity == 25
    assert pnl.complete is False


def test_short_pnl_direction_and_over_exit_validation() -> None:
    pnl = calculate_realized_pnl(
        [
            _leg(
                fill_id="short-entry",
                order_id="entry-order",
                kind="ENTRY",
                side="SELL",
                quantity=10,
                price=200.0,
            ),
            _leg(
                fill_id="short-exit",
                order_id="exit-order",
                kind="EXIT",
                side="BUY",
                quantity=10,
                price=190.0,
            ),
        ]
    )
    assert pnl.gross_pnl == 100.0
    with pytest.raises(FillValidationError):
        calculate_realized_pnl(
            [
                _leg(
                    fill_id="entry",
                    order_id="entry-order",
                    kind="ENTRY",
                    side="BUY",
                    quantity=10,
                    price=100.0,
                ),
                _leg(
                    fill_id="too-much",
                    order_id="exit-order",
                    kind="EXIT",
                    side="SELL",
                    quantity=11,
                    price=101.0,
                ),
            ]
        )


def test_sqlite_reopen_preserves_order_and_economics(tmp_path) -> None:
    path = tmp_path / "fills.db"
    store = BracketFillLedgerStore(path)
    legs = [
        _leg(
            fill_id="entry",
            order_id="entry-order",
            kind="ENTRY",
            side="BUY",
            quantity=65,
            price=100.0,
            recorded_at=1_700_000_001.0,
        ),
        _leg(
            fill_id="tp1",
            order_id="tp1-order",
            kind="EXIT",
            side="SELL",
            quantity=25,
            price=110.0,
            target="TP1",
            recorded_at=1_700_000_002.0,
        ),
    ]
    for leg in legs:
        assert store.record_fill(leg) is True
    reopened = BracketFillLedgerStore(path)
    restored = reopened.load_fills("entry-1")
    assert [leg.fill_id for leg in restored] == ["entry", "tp1"]
    assert [leg.economics() for leg in restored] == [leg.economics() for leg in legs]
    assert reopened.realized_pnl("entry-1").gross_pnl == 250.0
