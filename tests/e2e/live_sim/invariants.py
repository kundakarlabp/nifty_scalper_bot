from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InternalState:
    positions: dict[str, int] = field(default_factory=dict)
    active_stop: dict[str, int] = field(default_factory=dict)
    active_target: dict[str, int] = field(default_factory=dict)
    stop_prices: dict[str, list[float]] = field(default_factory=dict)
    risk_reserved: dict[str, float] = field(default_factory=dict)
    active_entries: dict[str, int] = field(default_factory=dict)


class TradingInvariantChecker:
    def __init__(self, broker, internal: InternalState | None = None) -> None:
        self.broker = broker
        self.internal = internal or InternalState()

    def check_all(self) -> None:
        self.assert_broker_internal_consistent()
        self.assert_entry_idempotency()
        self.assert_protection()
        self.assert_flat_cleanup()
        self.assert_bracket_quantity()
        self.assert_trailing_monotonicity()

    def assert_broker_internal_consistent(self) -> None:
        for symbol, broker_qty in self.broker.query_positions().items():
            internal_qty = self.internal.positions.get(symbol, 0)
            if internal_qty != broker_qty:
                raise AssertionError(
                    "broker/internal quantity mismatch for "
                    f"{symbol}: broker={broker_qty} internal={internal_qty}"
                )

    def assert_entry_idempotency(self) -> None:
        for symbol, count in self.internal.active_entries.items():
            if count > 1:
                raise AssertionError(
                    f"at most one active entry order allowed for {symbol}: {count}"
                )

    def assert_protection(self) -> None:
        for symbol, qty in self.internal.positions.items():
            if qty > 0 and not self.internal.active_stop.get(symbol):
                raise AssertionError(f"unprotected open position for {symbol}")

    def assert_bracket_quantity(self) -> None:
        for symbol, qty in self.internal.positions.items():
            stop_qty = self.internal.active_stop.get(symbol, 0)
            target_qty = self.internal.active_target.get(symbol, 0)
            if stop_qty > qty:
                raise AssertionError(
                    "stop quantity greater than position for "
                    f"{symbol}: {stop_qty}>{qty}"
                )
            if target_qty > qty:
                raise AssertionError(
                    "target quantity greater than position for "
                    f"{symbol}: {target_qty}>{qty}"
                )

    def assert_trailing_monotonicity(self) -> None:
        for symbol, prices in self.internal.stop_prices.items():
            if any(right < left for left, right in zip(prices, prices[1:])):
                raise AssertionError(f"stop loosening detected for {symbol}: {prices}")

    def assert_flat_cleanup(self) -> None:
        for symbol, qty in self.internal.positions.items():
            if qty == 0 and (
                self.internal.active_stop.get(symbol)
                or self.internal.active_target.get(symbol)
                or self.internal.risk_reserved.get(symbol)
            ):
                raise AssertionError(f"active bracket after flat for {symbol}")
