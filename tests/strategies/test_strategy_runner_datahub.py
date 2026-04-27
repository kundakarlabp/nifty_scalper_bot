from __future__ import annotations

from unittest.mock import MagicMock

from nifty_scalper_bot.strategies.runner import (
    StrategyRunner,
    StrategyRunnerConfig,
)


class _StubHub:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []

    def subscribe_ticks(
        self, symbol: str, callback
    ) -> None:  # pragma: no cover - simple
        self.subscribed.append(symbol)

    def unsubscribe_ticks(
        self, symbol: str, callback
    ) -> None:  # pragma: no cover - simple
        self.unsubscribed.append(symbol)


class _StubMDM:
    def __init__(self) -> None:
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.started = False
        self.degraded = False

    def subscribe(self, symbol: str, callback) -> None:
        self.subscribed.append(symbol)

    def unsubscribe(self, symbol: str, callback) -> None:
        self.unsubscribed.append(symbol)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def get_symbol_snapshot(self, symbol: str):
        class _Snapshot:
            def __init__(self, canonical_symbol: str, ltp: float | None) -> None:
                self.symbol = symbol
                self.canonical_symbol = canonical_symbol
                self.ltp = ltp
                self.bid = None
                self.ask = None
                self.mid = None
                self.tick_age_s = 1.0
                self.source = "ws"
                self.real_ticks_last_60s = 5
                self.latest_candle_provisional = False
                self.latest_candle_synthetic = False
                self.ohlc_valid = True

        if symbol in {"NIFTY", "NSE:NIFTY", "NIFTY 50"}:
            return _Snapshot("NSE:NIFTY", 25025.0)
        return _Snapshot(symbol, 100.0)

    def tracked_snapshot(self) -> list[str]:
        return [
            "NFO:NIFTY26MAY25000CE",
            "NFO:NIFTY26MAY25050CE",
            "NFO:NIFTY26MAY25100CE",
            "NFO:NIFTY26MAY25150CE",
            "NFO:NIFTY26MAY25200CE",
        ]


def _make_runner(hub: _StubHub | None) -> tuple[StrategyRunner, _StubMDM]:
    mdm = _StubMDM()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    position_manager = MagicMock()
    cfg = StrategyRunnerConfig(max_trade_history=10)
    runner = StrategyRunner(
        market_data_manager=mdm,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        config=cfg,
        data_hub=hub,
    )
    return runner, mdm


def test_runner_uses_hub_for_subscriptions() -> None:
    hub = _StubHub()
    runner, mdm = _make_runner(hub)

    runner.add_symbol("NIFTY25JUN25000CE")
    runner.start()

    assert hub.subscribed == ["NIFTY25JUN25000CE"]
    assert mdm.subscribed == []

    runner.stop()
    assert hub.unsubscribed == ["NIFTY25JUN25000CE"]


def test_runner_falls_back_to_mdm_when_hub_missing() -> None:
    runner, mdm = _make_runner(None)

    runner.add_symbol("NIFTY25SEP25000CE")
    runner.start()

    assert mdm.subscribed == ["NIFTY25SEP25000CE"]

    runner.stop()
    assert mdm.unsubscribed == ["NIFTY25SEP25000CE"]


def test_runner_avoids_message_bus_tick_subscription_when_hub_present() -> None:
    hub = _StubHub()
    mdm = _StubMDM()
    message_bus = MagicMock()
    runner = StrategyRunner(
        market_data_manager=mdm,
        indicator_engine=MagicMock(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(max_trade_history=10),
        data_hub=hub,
        message_bus=message_bus,
    )

    assert runner is not None
    message_bus.subscribe.assert_not_called()


def test_runner_subscribes_message_bus_tick_when_hub_missing() -> None:
    mdm = _StubMDM()
    message_bus = MagicMock()
    runner = StrategyRunner(
        market_data_manager=mdm,
        indicator_engine=MagicMock(),
        strategy_manager=MagicMock(),
        risk_manager=MagicMock(),
        order_manager=MagicMock(),
        position_manager=MagicMock(),
        config=StrategyRunnerConfig(max_trade_history=10),
        data_hub=None,
        message_bus=message_bus,
    )

    assert runner is not None
    message_bus.subscribe.assert_called_once()


def test_runner_start_skips_when_market_data_degraded() -> None:
    runner, mdm = _make_runner(None)
    mdm.degraded = True
    runner.add_symbol("NIFTY25SEP25000CE")

    runner.start()

    assert mdm.started is False
    assert mdm.subscribed == []


def test_build_candidate_snapshots_uses_canonical_spot() -> None:
    runner, _ = _make_runner(None)
    candidates = runner.build_candidate_snapshots(
        underlying="NIFTY 50", direction_bias="CE", window_each_side=2
    )
    assert candidates
    assert all(row["symbol"].startswith("NFO:NIFTY") for row in candidates)
    assert all("tick_age_s" in row for row in candidates)
