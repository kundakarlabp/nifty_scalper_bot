"""StrategyManager: lifecycle and performance tracking for all 12 strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Strategy


@dataclass
class StrategyPerf:
    signals: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0


class StrategyManager:
    def __init__(self, strategies: list["Strategy"]) -> None:
        self._strategies = {s.name: s for s in strategies}
        self._perf: dict[str, StrategyPerf] = {s.name: StrategyPerf() for s in strategies}

    def active_strategies(self) -> list["Strategy"]:
        return [s for s in self._strategies.values() if s.enabled]

    def get(self, name: str) -> "Strategy | None":
        return self._strategies.get(name)

    def enable(self, name: str) -> bool:
        s = self._strategies.get(name)
        if s:
            s._settings = type(s._settings)(**{**s._settings.model_dump(), "enabled": True})
            return True
        return False

    def disable(self, name: str) -> bool:
        s = self._strategies.get(name)
        if s:
            s._settings = type(s._settings)(**{**s._settings.model_dump(), "enabled": False})
            return True
        return False

    def record_outcome(self, strategy_name: str, is_win: bool, pnl: float) -> None:
        p = self._perf.get(strategy_name)
        if p:
            p.signals += 1
            p.total_pnl += pnl
            if is_win:
                p.wins += 1
            else:
                p.losses += 1

    def get_performance(self, name: str) -> dict:
        p = self._perf.get(name, StrategyPerf())
        return {
            "signals": p.signals,
            "win_rate": p.win_rate,
            "total_pnl": p.total_pnl,
            "wins": p.wins,
            "losses": p.losses,
        }

    def reset_all_daily(self) -> None:
        for s in self._strategies.values():
            s.reset_daily()

    def all_names(self) -> list[str]:
        return list(self._strategies.keys())
