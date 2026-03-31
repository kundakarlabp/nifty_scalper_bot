'''Tests for futures volume delta handling in signal generation.'''

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping

from nifty_scalper_bot.strategies.signal_generator import StrategyManager
from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


class StubIndicatorEngine:
    '''Stub indicator engine. Args: None. Returns: None. Raises: None.'''

    def update_price(
        self,
        symbol: str,
        price: Any,
        *,
        volume: int = 0,
        timestamp: Any | None = None,
    ) -> None:
        '''No-op update. Args: symbol, price, volume. Returns: None. Raises: None.'''
        try:
            return None
        except Exception as exc:
            logger.error('Failure in StubIndicatorEngine.update_price: %s', exc)
            raise

    def get_indicators(
        self, symbol: str, names: Iterable[str]
    ) -> Mapping[str, float | tuple[float, float, float] | None]:
        '''Return empty indicators. Args: symbol, names. Returns: map. Raises: None.'''
        try:
            return {}
        except Exception as exc:
            logger.error('Failure in StubIndicatorEngine.get_indicators: %s', exc)
            raise


class StubPositionManager:
    '''Stub position manager. Args: None. Returns: None. Raises: None.'''

    def get_position(self, symbol: str) -> None:
        '''Return no position. Args: symbol. Returns: None. Raises: None.'''
        try:
            return None
        except Exception as exc:
            logger.error('Failure in StubPositionManager.get_position: %s', exc)
            raise


class StubDataHub:
    '''Stub data hub for futures quotes. Args: volumes. Returns: None. Raises: None.'''

    def __init__(self, volumes: list[float]) -> None:
        '''Store futures volume samples. Args: volumes. Returns: None. Raises: None.'''
        try:
            self._volumes = deque(volumes)
            self._last_volume = volumes[-1] if volumes else 0.0
            self._index_calls = 0
        except Exception as exc:
            logger.error('Failure in StubDataHub.__init__: %s', exc)
            raise

    def get_quote(
        self, symbol: str, allow_pull: bool = False
    ) -> Mapping[str, Any] | None:
        '''Return a stub quote. Args: symbol, allow_pull. Returns: mapping or None. Raises: None.'''
        try:
            if symbol in {'NSE:NIFTY', 'NIFTY 50'}:
                self._index_calls += 1
                if self._index_calls == 1:
                    return {'last_price': 20000.0, 'vwap': 19950.0}
                return None
            if symbol == 'NIFTY':
                if self._volumes:
                    self._last_volume = self._volumes.popleft()
                return {'volume_traded_today': self._last_volume}
            return None
        except Exception as exc:
            logger.error('Failure in StubDataHub.get_quote: %s', exc)
            raise


def test_futures_volume_ratio_uses_delta() -> None:
    '''Validate futures ratio uses deltas. Args: None. Returns: None. Raises: None.'''
    try:
        data_hub = StubDataHub([100.0, 150.0])
        manager = StrategyManager(
            strategies=[],
            indicator_engine=StubIndicatorEngine(),
            position_manager=StubPositionManager(),
            data_hub=data_hub,
            futures_symbol='NIFTY',
        )
        indicators: dict[str, Any] = {}

        manager._augment_futures_metrics(indicators)
        manager._augment_futures_metrics(indicators)

        volume_ratio = indicators.get('futures_volume_ratio')
        assert volume_ratio is not None
        assert 0.6 < float(volume_ratio) < 0.8
    except Exception as exc:
        logger.error('Failure in test_futures_volume_ratio_uses_delta: %s', exc)
        raise


def test_index_cache_used_when_quote_missing() -> None:
    '''Validate index cache fills missing quotes. Args: None. Returns: None. Raises: None.'''
    try:
        data_hub = StubDataHub([100.0])
        manager = StrategyManager(
            strategies=[],
            indicator_engine=StubIndicatorEngine(),
            position_manager=StubPositionManager(),
            data_hub=data_hub,
            futures_symbol='NIFTY',
        )

        first_indicators: dict[str, Any] = {}
        manager._augment_futures_metrics(first_indicators)

        second_indicators: dict[str, Any] = {}
        manager._augment_futures_metrics(second_indicators)

        assert second_indicators.get('nifty_index_ltp') == 20000.0
        assert second_indicators.get('nifty_index_vwap') == 19950.0
    except Exception as exc:
        logger.error('Failure in test_index_cache_used_when_quote_missing: %s', exc)
        raise
