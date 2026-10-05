"""Canonical structural setup lifecycle observability.

Strategies own setup semantics; StrategyManager owns arbitration; Runner owns
structural execution validation; execution owners remain unchanged. This module
owns only the shared, bounded lifecycle record keyed by structural setup identity.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SetupStage(StrEnum):
    ARMED = "SETUP_ARMED"
    CONFIRMING = "CONFIRMING"
    TRIGGER_QUALIFIED = "TRIGGER_QUALIFIED"
    MANAGER_QUALIFIED = "MANAGER_QUALIFIED"
    RUNNER_APPROVED = "RUNNER_APPROVED"
    TRADE_PLAN = "TRADE_PLAN"
    BROKER_ACCEPTED = "BROKER_ACCEPTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    CONTEXT_VETOED = "CONTEXT_VETOED"
    CONTRACT_REJECTED = "CONTRACT_REJECTED"
    RISK_REJECTED = "RISK_REJECTED"
    EXECUTION_REJECTED = "EXECUTION_REJECTED"


_TERMINAL = {
    SetupStage.BROKER_ACCEPTED,
    SetupStage.INVALIDATED,
    SetupStage.EXPIRED,
    SetupStage.CONTEXT_VETOED,
    SetupStage.CONTRACT_REJECTED,
    SetupStage.RISK_REJECTED,
    SetupStage.EXECUTION_REJECTED,
}
_RANK = {
    SetupStage.ARMED: 10,
    SetupStage.CONFIRMING: 20,
    SetupStage.TRIGGER_QUALIFIED: 30,
    SetupStage.MANAGER_QUALIFIED: 40,
    SetupStage.RUNNER_APPROVED: 50,
    SetupStage.TRADE_PLAN: 60,
    SetupStage.BROKER_ACCEPTED: 70,
}


@dataclass(slots=True)
class SetupLifecycleRecord:
    key: str
    setup_id: str
    strategy: str
    symbol: str
    side: str
    stage: str
    reason: str | None
    first_seen_ts: float
    updated_ts: float
    transitions: int = 1


class SetupLifecycleRegistry:
    """Bounded in-process SSOT for setup-stage observability only."""

    def __init__(self, limit: int = 2048) -> None:
        self._limit = max(128, int(limit))
        self._lock = threading.RLock()
        self._records: OrderedDict[str, SetupLifecycleRecord] = OrderedDict()
        self._transition_counts: Counter[str] = Counter()

    @staticmethod
    def key(strategy: object, setup_id: object, side: object = "") -> str:
        strategy_key = str(strategy or "unknown").strip().lower()
        side_key = str(side or "").strip().upper()
        setup_key = str(setup_id or "").strip()
        return f"{strategy_key}:{side_key}:{setup_key}"

    def transition(
        self,
        stage: SetupStage | str,
        *,
        strategy: object,
        setup_id: object,
        symbol: object = "",
        side: object = "",
        reason: object = None,
    ) -> SetupLifecycleRecord | None:
        setup = str(setup_id or "").strip()
        if not setup:
            return None
        resolved = SetupStage(stage)
        key = self.key(strategy, setup, side)
        now = time.time()
        with self._lock:
            previous = self._records.get(key)
            if previous is not None:
                previous_stage = SetupStage(previous.stage)
                if previous_stage in _TERMINAL:
                    return previous
                previous_rank = _RANK.get(previous_stage, 0)
                next_rank = _RANK.get(resolved, previous_rank + 1)
                if resolved not in _TERMINAL and next_rank < previous_rank:
                    return previous
                same_reason = str(reason or "") == str(previous.reason or "")
                if previous_stage == resolved and same_reason:
                    previous.updated_ts = now
                    self._records.move_to_end(key)
                    return previous
                first_seen = previous.first_seen_ts
                transitions = previous.transitions + 1
            else:
                first_seen = now
                transitions = 1
            record = SetupLifecycleRecord(
                key=key,
                setup_id=setup,
                strategy=str(strategy or "unknown"),
                symbol=str(symbol or ""),
                side=str(side or "").upper(),
                stage=resolved.value,
                reason=str(reason) if reason not in (None, "") else None,
                first_seen_ts=first_seen,
                updated_ts=now,
                transitions=transitions,
            )
            self._records[key] = record
            self._records.move_to_end(key)
            self._transition_counts[resolved.value] += 1
            while len(self._records) > self._limit:
                self._records.popitem(last=False)
        LOGGER.info(
            (
                "SETUP_LIFECYCLE stage=%s strategy=%s symbol=%s side=%s "
                "setup_id=%s reason=%s"
            ),
            resolved.value,
            record.strategy,
            record.symbol,
            record.side,
            record.setup_id,
            record.reason,
            extra={
                "event": "SETUP_LIFECYCLE",
                "setup_stage": resolved.value,
                "strategy": record.strategy,
                "symbol": record.symbol,
                "side": record.side,
                "setup_id": record.setup_id,
                "reason": record.reason,
            },
        )
        return record

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = [
                asdict(item)
                for item in self._records.values()
                if SetupStage(item.stage) not in _TERMINAL
            ]
            terminal = Counter(
                item.stage
                for item in self._records.values()
                if SetupStage(item.stage) in _TERMINAL
            )
            return {
                "active_count": len(active),
                "active": active,
                "terminal_counts": dict(terminal),
                "transition_counts": dict(self._transition_counts),
                "tracked_count": len(self._records),
            }


SETUP_LIFECYCLE = SetupLifecycleRegistry()


def transition_setup(
    stage: SetupStage | str,
    metadata: Mapping[str, Any] | None = None,
    *,
    strategy: object = "",
    setup_id: object = "",
    symbol: object = "",
    side: object = "",
    reason: object = None,
) -> SetupLifecycleRecord | None:
    payload = dict(metadata or {})
    return SETUP_LIFECYCLE.transition(
        stage,
        strategy=strategy or payload.get("strategy_name") or payload.get("strategy"),
        setup_id=(
            setup_id or payload.get("setup_id") or payload.get("setup_structure_id")
        ),
        symbol=symbol or payload.get("candidate_symbol") or payload.get("symbol"),
        side=(
            side
            or payload.get("contract_side")
            or payload.get("trade_side")
            or payload.get("side")
        ),
        reason=reason,
    )


__all__ = [
    "SETUP_LIFECYCLE",
    "SetupLifecycleRecord",
    "SetupLifecycleRegistry",
    "SetupStage",
    "transition_setup",
]
