"""Execution stress scenarios for deterministic exit behavior."""

from __future__ import annotations

from dataclasses import dataclass

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class _DummyOrderManager:
    exits: list[tuple[str, int]]


class _Harness:
    def __init__(self) -> None:
        """Args: none; Returns: None; Raises: None."""
        self._exits: list[tuple[str, int]] = []
        self._manager = BracketManager(order_manager=_DummyOrderManager(self._exits))
        self._manager.attach_exit_executor(self._capture_exit)

    def _capture_exit(self, symbol: str, qty: int) -> None:
        """Args: symbol, qty; Returns: None; Raises: None."""
        self._exits.append((symbol, qty))

    def run_ticks(self, symbol: str, ticks: list[float]) -> None:
        """Args: symbol, ticks; Returns: None; Raises: None."""
        for price in ticks:
            self._manager.on_tick_event({"symbol": symbol, "ltp": price})

    @property
    def exits(self) -> list[tuple[str, int]]:
        """Args: none; Returns: list; Raises: None."""
        return list(self._exits)


def scenario_sl_gap() -> bool:
    """Args: none; Returns: bool; Raises: None."""
    h = _Harness()
    h._manager.register_virtual_bracket(
        "sl-gap", "NIFTY", "BUY", 1, 200.0, 198.0, 220.0
    )
    h.run_ticks("NIFTY", [200.0, 198.0, 195.0, 190.0, 185.0])
    ok = len(h.exits) > 0
    LOGGER.info("SL_TRIGGERED scenario_sl_gap=%s", ok)
    return ok


def scenario_fast_reversal() -> bool:
    """Args: none; Returns: bool; Raises: None."""
    h = _Harness()
    h._manager.register_virtual_bracket("rev", "NIFTY", "BUY", 1, 200.0, 195.0, 210.0)
    h.run_ticks("NIFTY", [200.0, 209.0, 203.0, 196.0, 194.0])
    ok = len(h.exits) > 0
    LOGGER.info("EXIT_EXECUTED scenario_fast_reversal=%s", ok)
    return ok


def scenario_trailing_profit_lock() -> bool:
    """Args: none; Returns: bool; Raises: None."""
    h = _Harness()
    h._manager.register_virtual_bracket("trail", "NIFTY", "BUY", 1, 200.0, 190.0, 240.0)
    h.run_ticks("NIFTY", [200.0, 210.0, 220.0, 215.0])
    state = h._manager._brackets.get("trail")
    ok = bool(state and state.highest_ltp >= 220.0)
    LOGGER.info("PNL_UPDATE scenario_trailing_profit_lock=%s", ok)
    return ok


def scenario_tp_spike() -> bool:
    """Args: none; Returns: bool; Raises: None."""
    h = _Harness()
    h._manager.register_virtual_bracket("tp", "NIFTY", "BUY", 1, 200.0, 190.0, 205.0)
    h.run_ticks("NIFTY", [200.0, 206.0])
    ok = len(h.exits) > 0
    LOGGER.info("EXIT_EXECUTED scenario_tp_spike=%s", ok)
    return ok


def main() -> None:
    """Args: none; Returns: None; Raises: None."""
    results = [
        scenario_sl_gap(),
        scenario_fast_reversal(),
        scenario_trailing_profit_lock(),
        scenario_tp_spike(),
    ]
    passed = sum(1 for result in results if result)
    LOGGER.info(
        "Condition met: stress_complete passed=%s total=%s",
        passed,
        len(results),
    )


if __name__ == "__main__":
    main()


def test_sl_gap() -> None:
    """Args: none; Returns: None; Raises: AssertionError."""
    assert scenario_sl_gap()
