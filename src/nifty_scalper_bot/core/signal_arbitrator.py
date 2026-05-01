"""Signal arbitration and deterministic trade decision engine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class _SymbolState:
    direction: str
    last_ts: float


@dataclass(slots=True, frozen=True)
class StrategyVote:
    strategy: str
    direction: str
    score: float
    confidence: float
    reasons: list[str]
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TradeDecision:
    action: str
    direction: str
    symbol: str | None
    underlying: str
    score: float
    confidence: float
    entry_price: float | None
    stop_loss: float | None
    target: float | None
    rr: float | None
    reasons: list[str]
    votes: list[StrategyVote]
    candidate_meta: dict[str, Any]
    trace_id: str
    timestamp: float


class SignalArbitrator:
    def __init__(self, cooldown_seconds: float = 3.0) -> None:
        self._cooldown_seconds = max(float(cooldown_seconds), 0.0)
        self._active_symbols: set[str] = set()
        self._state: dict[str, _SymbolState] = {}
        self._lock = threading.RLock()

    def allow(self, signal: Any, action: str | None = None) -> bool:
        if isinstance(signal, str):
            symbol = str(signal or '').upper()
            normalized_action = str(action or '').upper()
        else:
            symbol = str(getattr(signal, 'symbol', '') or '').upper()
            normalized_action = str(getattr(signal, 'action', action or '') or '').upper()
        if not symbol:
            return False
        direction = self._direction(normalized_action)
        now = time.time()
        with self._lock:
            if symbol in self._active_symbols:
                return False
            prev = self._state.get(symbol)
            if prev is None:
                return True
            if direction and prev.direction == direction:
                return False
            return now - prev.last_ts >= self._cooldown_seconds

    def register(self, symbol: str, action: str = '') -> None:
        key = str(symbol or '').upper()
        if not key:
            return
        with self._lock:
            self._active_symbols.add(key)
            self._state[key] = _SymbolState(direction=self._direction(action), last_ts=time.time())

    def release(self, symbol: str) -> None:
        key = str(symbol or '').upper()
        if not key:
            return
        with self._lock:
            self._active_symbols.discard(key)

    def decide(self, *, underlying: str, votes: list[StrategyVote], option_candidates: list[dict[str, Any]], market_context: dict[str, Any], trace_id: str) -> TradeDecision:
        mode = str(market_context.get('quality_mode') or 'normal').lower()
        min_score = {'strict': 7.5, 'normal': 6.5, 'loose': 5.8}.get(mode, 6.5)
        min_conf = {'strict': 0.75, 'normal': 0.65, 'loose': 0.58}.get(mode, 0.65)
        min_votes = {'strict': 3, 'normal': 2, 'loose': 2}.get(mode, 2)
        if market_context.get('regime_trade_allowed') is False:
            return self._hold(underlying, votes, trace_id, 'regime_block')
        if market_context.get('spot_stale') is True:
            return self._hold(underlying, votes, trace_id, 'spot_stale')

        agg = {'CE': [], 'PE': []}
        for v in votes:
            d = str(v.direction).upper()
            if d not in agg:
                continue
            score = max(0.0, min(10.0, float(v.score)))
            conf = float(v.confidence)
            conf = conf / 10.0 if conf > 1.0 else conf
            conf = max(0.0, min(1.0, conf))
            w = self._weight(v.strategy)
            agg[d].append((score * w, conf, v))

        side_stats = {}
        for side, arr in agg.items():
            if not arr:
                side_stats[side] = (0.0, 0.0, 0)
                continue
            wt_score = sum(x[0] for x in arr) / max(1.0, sum(self._weight(x[2].strategy) for x in arr))
            conf = sum(x[1] for x in arr) / len(arr)
            side_stats[side] = (wt_score, conf, len(arr))

        ce_score, ce_conf, ce_n = side_stats['CE']
        pe_score, pe_conf, pe_n = side_stats['PE']
        if ce_score >= min_score and pe_score >= min_score and abs(ce_score - pe_score) < 1.2:
            return self._hold(underlying, votes, trace_id, 'conflicting_direction')
        side = 'CE' if ce_score >= pe_score else 'PE'
        score, conf, count = side_stats[side]
        if count < min_votes:
            return self._hold(underlying, votes, trace_id, 'insufficient_votes')
        if score < min_score:
            return self._hold(underlying, votes, trace_id, 'score_low')
        if conf < min_conf:
            return self._hold(underlying, votes, trace_id, 'confidence_low')
        candidates = [c for c in option_candidates if str(c.get('side', '')).upper() == side]
        if not candidates:
            return self._hold(underlying, votes, trace_id, 'no_valid_option_candidate')
        best = max(candidates, key=lambda c: float(c.get('final_score', c.get('score', 0.0)) or 0.0))
        rr = float(best.get('rr') or 0.0)
        if rr < 1.5:
            return self._hold(underlying, votes, trace_id, 'rr_low')
        return TradeDecision('BUY', side, str(best.get('symbol')), underlying, round(score, 3), round(conf, 3), best.get('entry_price'), best.get('stop_loss'), best.get('target'), rr, ['aligned_votes'], votes, dict(best), trace_id, time.time())

    def _hold(self, underlying: str, votes: list[StrategyVote], trace_id: str, reason: str) -> TradeDecision:
        return TradeDecision('HOLD', 'NONE', None, underlying, 0.0, 0.0, None, None, None, None, [reason], votes, {}, trace_id, time.time())

    @staticmethod
    def _weight(name: str) -> float:
        n = name.lower()
        weights = {
            'smc': 1.3, 'liquidity': 1.3, 'order block': 1.3, 'vwap': 1.2,
            'orb': 1.15, 'cpr': 1.1, 'rsi': 0.95, 'bb': 1.0, 'futures': 1.2,
            'regime': 1.2, 'microstructure': 1.3,
        }
        for k, v in weights.items():
            if k in n:
                return v
        return 1.0

    @staticmethod
    def _direction(action: str) -> str:
        if action in {'BUY', 'CLOSE_SHORT'}:
            return 'LONG'
        if action in {'SELL', 'CLOSE_LONG'}:
            return 'SHORT'
        return ''
