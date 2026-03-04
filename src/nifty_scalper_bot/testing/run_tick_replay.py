"""CLI entrypoint for deterministic replay against the engine."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.core.strategy_manager import StrategyManager
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.order_manager import OrderManager
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.execution.risk_manager import RiskManager
from nifty_scalper_bot.testing.simulated_broker import SimulatedZerodhaBroker
from nifty_scalper_bot.testing.tick_replay_engine import TickReplayEngine
from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.rate_limiter import RateLimiter

LOGGER = get_logger(__name__)


class _DummyMarketData:
    def attach_tick_bus(self, bus: Any) -> None:
        """Args: bus; Returns: None; Raises: None."""


@dataclass(slots=True)
class ReplayOrchestrator:
    """Route replay ticks through data and execution subsystems."""

    data_hub: DataHub
    strategy_manager: StrategyManager
    order_manager: OrderManager
    bracket_manager: BracketManager
    risk_manager: RiskManager
    broker: SimulatedZerodhaBroker

    def on_tick_event(self, tick: dict[str, Any]) -> None:
        """Args: tick; Returns: None; Raises: None."""
        try:
            symbol = str(tick.get("symbol") or "")
            ltp = float(tick.get("ltp") or 0.0)
            if not symbol or ltp <= 0:
                return
            self.broker.set_price(symbol, ltp)
            self.data_hub.ingest_tick_sync({"symbol": symbol, "ltp": ltp})
            self.order_manager.on_tick_event({"symbol": symbol, "ltp": ltp})
            self.bracket_manager.on_tick_event({"symbol": symbol, "ltp": ltp})
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failure in ReplayOrchestrator.on_tick_event: %s", exc)


def build_orchestrator() -> ReplayOrchestrator:
    """Args: none; Returns: ReplayOrchestrator; Raises: Exception."""
    try:
        broker = SimulatedZerodhaBroker()
        data_hub = DataHub(_DummyMarketData(), instrument_resolver=None)
        strategy_manager = StrategyManager(
            [],
            indicator_engine=None,
            position_manager=None,
        )
        order_manager = OrderManager(
            broker_client=broker,
            position_manager=PositionManager(),
            rate_limiter=RateLimiter(),
            instrument_resolver=None,
        )
        bracket_manager = BracketManager(order_manager=order_manager)
        risk_manager = RiskManager(
            risk_config=RiskConfig(),
            position_manager=PositionManager(),
            initial_balance=100000.0,
        )
        return ReplayOrchestrator(
            data_hub=data_hub,
            strategy_manager=strategy_manager,
            order_manager=order_manager,
            bracket_manager=bracket_manager,
            risk_manager=risk_manager,
            broker=broker,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failure in build_orchestrator: %s", exc)
        raise


def main() -> None:
    """Args: none; Returns: None; Raises: Exception."""
    parser = argparse.ArgumentParser(
        description="Deterministic tick replay runner",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="CSV file path with symbol,ltp,timestamp",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable wall-clock delays and replay ticks deterministically",
    )
    args = parser.parse_args()
    engine = TickReplayEngine(tick_file=args.file, speed=args.speed)
    engine.load()
    orchestrator = build_orchestrator()
    engine.replay(orchestrator.on_tick_event, deterministic=args.deterministic)
    LOGGER.info("Condition met: replay_complete ticks=%s", engine.index)


if __name__ == "__main__":
    main()
