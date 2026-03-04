"""Strategy and execution validation harness."""

from __future__ import annotations

import os
from dataclasses import dataclass

from nifty_scalper_bot.testing.simulated_broker import SimulatedZerodhaBroker
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class _SimpleStrategy:
    symbol: str

    def generate_signal(
        self,
        symbol: str,
        current_price: float,
    ) -> dict[str, object] | None:
        """Args: symbol, current_price; Returns: dict|None; Raises: None."""
        if symbol != self.symbol:
            return None
        side = 'BUY' if current_price >= 200.0 else 'SELL'
        return {
            'symbol': symbol,
            'side': side,
            'confidence': 0.8,
            'reason': 'test_signal',
        }


def run_strategy_validation() -> dict[str, float | int]:
    """Args: none; Returns: dict; Raises: None."""
    broker = SimulatedZerodhaBroker()
    mode = str(os.getenv('EXECUTION_MODE', 'PAPER')).strip().upper()
    strategy = _SimpleStrategy(symbol='NIFTY')
    ticks = [198.0, 201.0, 204.0, 199.0]
    signals = 0
    entries = 0
    exits = 0
    pnl = 0.0
    entry_price = None
    for ltp in ticks:
        broker.set_price('NIFTY', ltp)
        signal = strategy.generate_signal('NIFTY', ltp)
        if signal is None:
            continue
        signals += 1
        if str(signal.get('side')) == 'BUY' and entry_price is None:
            if mode == 'SHADOW':
                LOGGER.info('WOULD_BUY symbol=NIFTY qty=1')
            else:
                broker.place_order(
                    symbol='NIFTY',
                    side='BUY',
                    quantity=1,
                    order_type='MARKET',
                )
            entry_price = ltp
            entries += 1
        elif str(signal.get('side')) == 'SELL' and entry_price is not None:
            if mode == 'SHADOW':
                LOGGER.info('WOULD_EXIT symbol=NIFTY qty=1')
            else:
                broker.place_order(
                    symbol='NIFTY',
                    side='SELL',
                    quantity=1,
                    order_type='MARKET',
                )
            pnl += ltp - entry_price
            exits += 1
            entry_price = None
    LOGGER.info(
        'signals generated=%s entries=%s exits=%s profit=%s',
        signals,
        entries,
        exits,
        round(pnl, 2),
    )
    return {
        'signals': signals,
        'entries': entries,
        'exits': exits,
        'profit': round(pnl, 2),
    }


def main() -> None:
    """Args: none; Returns: None; Raises: None."""
    result = run_strategy_validation()
    print(
        'signals generated={signals} entries={entries} exits={exits} '
        'profit={profit}'.format(**result)
    )


if __name__ == '__main__':
    main()
