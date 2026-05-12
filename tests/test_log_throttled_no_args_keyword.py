"""Regression tests for log_throttled usage in live signal path."""

from __future__ import annotations

import typing as t
from pathlib import Path

from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _NoopStrategy:
    """Strategy stub that returns no signal. Args: none. Returns: none. Raises: none."""

    name = 'VWAPPro'

    def get_required_indicators(self) -> list[str]:
        """Return required indicators. Args: none. Returns: list[str]. Raises: none."""
        return ['vwap', 'exchange_vwap', 'volume', 'avg_volume']

    def generate_signal(
        self,
        symbol: str,
        indicators: t.Mapping[str, t.Any],
        current_price: float,
        position: t.Any,
    ) -> None:
        """Return no signal. Args: symbol/indicators/current_price/position. Returns: None. Raises: none."""
        return None


class _Engine:
    """Minimal indicator engine. Args: none. Returns: none. Raises: none."""

    def get_indicators(self, symbol: str, names: t.Iterable[str]) -> dict[str, float]:
        """Return fixed indicator values. Args: symbol/names. Returns: dict[str,float]. Raises: none."""
        _ = symbol
        values = {str(name): 1.0 for name in names}
        values['vwap'] = 100.0
        values['exchange_vwap'] = 100.0
        values['volume'] = 10_000.0
        values['avg_volume'] = 8_000.0
        return values


class _Pos:
    """Position manager stub. Args: none. Returns: none. Raises: none."""

    def get_position(self, symbol: str) -> None:
        """Return no position. Args: symbol. Returns: None. Raises: none."""
        _ = symbol
        return None


def test_no_log_throttled_args_keyword_in_source() -> None:
    """Ensure no invalid args keyword is passed to log_throttled. Args: none. Returns: none. Raises: none."""
    src_root = Path(__file__).resolve().parents[1] / 'src'
    offenders: list[str] = []
    for py_path in src_root.rglob('*.py'):
        text = py_path.read_text(encoding='utf-8')
        if 'log_throttled(' in text and 'args=' in text:
            offenders.append(str(py_path.relative_to(src_root.parent)))
    assert offenders == []


def test_strategy_manager_generate_signal_no_log_typeerror() -> None:
    """Ensure StrategyManager.generate_signal does not raise log_throttled TypeError. Args: none. Returns: none. Raises: none."""
    manager = StrategyManager([_NoopStrategy()], _Engine(), _Pos())
    signal = manager.generate_signal('NSE:NIFTY', 100.0)
    assert signal is None
