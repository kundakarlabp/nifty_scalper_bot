"""Deterministic scalping strategy using weighted indicator scoring."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.indicators import WeightedSignalInputs, weighted_score
from nifty_scalper_bot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class ScalpingStrategyConfig:
    """Strategy scoring configuration. Args: threshold. Returns: config. Raises: None."""

    score_threshold: float

    @classmethod
    def from_env(cls) -> "ScalpingStrategyConfig":
        """Load strategy config from environment. Args: none. Returns: config. Raises: ValueError."""
        raw = os.getenv("SCALPING_SCORE_THRESHOLD", "2.5")
        return cls(score_threshold=abs(float(raw)))


class ScalpingStrategy:
    """Build BUY/SELL/HOLD signals from weighted indicator scores."""

    def __init__(self, config: ScalpingStrategyConfig | None = None) -> None:
        """Initialise strategy. Args: optional config. Returns: None. Raises: None."""
        self.config = config or ScalpingStrategyConfig.from_env()

    def evaluate(
        self, symbol: str, price: float, indicators: dict[str, Any]
    ) -> Signal | None:
        """Evaluate one closed candle. Args: symbol/price/indicators. Returns: signal or None. Raises: KeyError."""
        values = WeightedSignalInputs(
            ema=float(indicators["ema"]),
            rsi=float(indicators["rsi"]),
            macd=float(indicators["macd"]),
            vwap=float(indicators["vwap"]),
            adx=float(indicators["adx"]),
        )
        score = weighted_score(values)
        logger.info(
            '{"event":"SIGNAL_GENERATED","symbol":"%s","score":%.4f}', symbol, score
        )
        threshold = abs(self.config.score_threshold)
        if abs(score) < threshold:
            logger.info(
                '{"event":"SIGNAL_REJECTED","symbol":"%s","reason":"score_below_threshold","score":%.4f,"threshold":%.4f}',
                symbol,
                score,
                threshold,
            )
            return None
        action = "BUY" if score >= 0 else "SELL"
        logger.info(
            '{"event":"SIGNAL_EXECUTED","symbol":"%s","action":"%s","score":%.4f}',
            symbol,
            action,
            score,
        )
        return Signal(
            action=action,
            symbol=symbol,
            quantity=1,
            confidence=min(1.0, abs(score) / max(1.0, threshold)),
            reason="weighted_score",
            stop_loss=None,
            take_profit=None,
            metadata={"score": score},
        )
