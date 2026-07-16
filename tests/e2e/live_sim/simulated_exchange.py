from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Instrument:
    symbol: str
    token: int
    exchange: str
    instrument_type: str
    strike: float | None
    expiry: str | None
    lot_size: int
    tick_size: float


class SimulatedExchange:
    def __init__(self, clock, broker=None, recorder=None) -> None:
        self.clock = clock
        self.broker = broker
        self.recorder = recorder
        self.instruments: dict[str, Instrument] = {}
        self.subscriptions: dict[str, int] = {}
        self.generation = 1
        self.connected = True
        self._seq = 0
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._pending = 0

    def add_instrument(self, instrument: Instrument) -> None:
        self.instruments[instrument.symbol] = instrument

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def confirm_subscription(self, symbol: str) -> None:
        self.subscriptions[symbol] = self.generation
        if self.recorder:
            self.recorder.record(
                "SUBSCRIPTION_CONFIRMED", symbol, generation=self.generation
            )

    def rotate_subscription_generation(self) -> int:
        self.generation += 1
        return self.generation

    def disconnect_feed(self) -> None:
        self.connected = False

    def reconnect_feed(self) -> None:
        self.connected = True
        self.rotate_subscription_generation()

    def publish_tick(
        self,
        symbol: str,
        *,
        ltp: float,
        bid: float | None = None,
        ask: float | None = None,
        volume: int = 1,
        timestamp=None,
        **extra: Any,
    ) -> dict[str, Any]:
        if not self.connected:
            return {}
        inst = self.instruments[symbol]
        self._seq += 1
        tick = {
            "symbol": symbol,
            "token": inst.token,
            "instrument_token": inst.token,
            "timestamp": timestamp or self.clock.now(),
            "exchange_timestamp": timestamp or self.clock.now(),
            "received_timestamp": self.clock.now(),
            "ltp": ltp,
            "last_price": ltp,
            "bid": bid if bid is not None else ltp,
            "ask": ask if ask is not None else ltp,
            "bid_quantity": 100,
            "ask_quantity": 100,
            "depth": extra.get("depth", {}),
            "volume": volume,
            "subscription_generation": self.subscriptions.get(symbol, self.generation),
            "sequence": self._seq,
        }
        self._pending += 1
        for subscriber in list(self._subscribers):
            subscriber(tick)
        if self.broker:
            self.broker.on_quote(symbol, bid=tick["bid"], ask=tick["ask"], ltp=ltp)
        self._pending -= 1
        return tick

    def publish_quote(self, symbol: str, **kwargs: Any) -> dict[str, Any]:
        return self.publish_tick(symbol, **kwargs)

    def publish_depth(
        self, symbol: str, *, bid: float, ask: float, **kwargs: Any
    ) -> dict[str, Any]:
        return self.publish_tick(
            symbol, bid=bid, ask=ask, ltp=kwargs.pop("ltp", (bid + ask) / 2), **kwargs
        )

    def publish_minute_path(self, symbol: str, prices: list[float]) -> None:
        for price in prices:
            self.publish_tick(symbol, ltp=price, bid=price - 0.2, ask=price)

    def publish_batch(self, ticks: list[dict[str, Any]]) -> None:
        for tick in ticks:
            self.publish_tick(**tick)

    @property
    def pending_events(self) -> int:
        return self._pending
