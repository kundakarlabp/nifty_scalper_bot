from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

logger = logging.getLogger(__name__)
Side = Literal["BUY", "SELL"]


@dataclass(slots=True)
class Signal:
    """
    Typed representation of a trading signal (immutable intent, mutable rounding/hash).

    Fields:
      signal: "BUY" | "SELL"
      score: integer score used by filters
      confidence: 0..10 (clamped)
      entry_price, stop_loss, target: absolute prices (>0)
      reasons: human-readable reasons that formed the signal
      market_volatility: usually ATR (optional)
      hash: stable identifier set by compute_hash()
    """
    signal: Side
    score: int
    confidence: float
    entry_price: float
    stop_loss: float
    target: float
    reasons: List[str] = field(default_factory=list)
    market_volatility: Optional[float] = None
    hash: Optional[str] = None  # set by compute_hash()

    # ------------------------- validation & normalization -------------------------

    def __post_init__(self) -> None:
        # Normalize side
        s = str(self.signal).upper()
        if s not in ("BUY", "SELL"):
            raise ValueError(f"signal must be 'BUY' or 'SELL', got {self.signal!r}")
        object.__setattr__(self, "signal", s)

        # Coerce numerics
        try:
            ep = float(self.entry_price)
            sl = float(self.stop_loss)
            tp = float(self.target)
        except Exception as e:
            raise ValueError(f"entry/stop/target must be numeric: {e}") from e

        if ep <= 0 or sl <= 0 or tp <= 0:
            raise ValueError("entry_price, stop_loss, target must be > 0")

        # Stop distance must be non-zero
        if ep == sl:
            raise ValueError("stop_loss cannot equal entry_price")

        # Directional sanity check
        if s == "BUY" and tp <= ep:
            logger.warning("BUY signal has non-positive reward: target %.2f <= entry %.2f", tp, ep)
        if s == "SELL" and tp >= ep:
            logger.warning("SELL signal has non-positive reward: target %.2f >= entry %.2f", tp, ep)

        # Clamp confidence into 0..10
        try:
            conf = float(self.confidence)
        except Exception:
            conf = 0.0
        conf = max(0.0, min(10.0, conf))
        object.__setattr__(self, "confidence", conf)

        # Score as int
        object.__setattr__(self, "score", int(self.score))

        # Clean reasons
        if self.reasons is None:
            object.__setattr__(self, "reasons", [])
        else:
            cleaned = [str(r) for r in self.reasons if r is not None]
            object.__setattr__(self, "reasons", cleaned)

    # ------------------------- convenience properties -------------------------

    @property
    def is_buy(self) -> bool:
        return self.signal == "BUY"

    @property
    def is_sell(self) -> bool:
        return self.signal == "SELL"

    @property
    def r_points(self) -> float:
        """Absolute stop distance in price points."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward_r(self) -> float:
        """Reward in R multiples (|target-entry| / |entry-stop|)."""
        r = self.r_points
        if r <= 0:
            return 0.0
        return abs(self.target - self.entry_price) / r

    # ------------------------- utils -------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Dict payload compatible with Telegram / executors."""
        return {
            "side": self.signal,  # alias for compatibility
            "signal": self.signal,
            "score": int(self.score),
            "confidence": float(self.confidence),
            "entry_price": float(self.entry_price),
            "stop_loss": float(self.stop_loss),
            "target": float(self.target),
            "reasons": list(self.reasons),
            "market_volatility": None if self.market_volatility is None else float(self.market_volatility),
            "hash": self.hash,
        }

    def to_json(self) -> str:
        """Compact JSON representation (useful for logging/telemetry)."""
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    def compute_hash(self) -> str:
        """
        Stable hash over core economic fields (side, entry, sl, tp, score, confidence).
        Saves into self.hash and returns it.
        """
        payload = {
            "signal": self.signal,
            "entry": round(self.entry_price, 6),
            "sl": round(self.stop_loss, 6),
            "tp": round(self.target, 6),
            "score": int(self.score),
            "conf": round(self.confidence, 3),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        h = hashlib.blake2b(raw, digest_size=12).hexdigest()
        object.__setattr__(self, "hash", h)
        return h

    def round_prices_inplace(self, tick_size: float) -> None:
        """Round entry/SL/TP to the nearest exchange tick (mutates in place)."""
        try:
            t = float(tick_size)
        except Exception:
            return
        if t <= 0:
            return

        def _rt(x: float) -> float:
            return round(float(x) / t) * t

        ep = _rt(self.entry_price)
        sl = _rt(self.stop_loss)
        tp = _rt(self.target)

        # Avoid zero stop distance due to rounding
        if ep == sl:
            if self.is_buy:
                sl = max(t, sl - t) if sl > t else sl + t
            else:
                sl = sl + t

        object.__setattr__(self, "entry_price", ep)
        object.__setattr__(self, "stop_loss", sl)
        object.__setattr__(self, "target", tp)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Signal":
        """Lenient constructor from a dict."""
        return cls(
            signal=str(d.get("signal", d.get("side", "BUY"))).upper(),
            score=int(d.get("score", 0)),
            confidence=float(d.get("confidence", 0.0)),
            entry_price=float(d.get("entry_price")),
            stop_loss=float(d.get("stop_loss")),
            target=float(d.get("target")),
            reasons=list(d.get("reasons", [])) if d.get("reasons") is not None else [],
            market_volatility=(None if d.get("market_volatility") is None else float(d.get("market_volatility"))),
            hash=d.get("hash"),
        )