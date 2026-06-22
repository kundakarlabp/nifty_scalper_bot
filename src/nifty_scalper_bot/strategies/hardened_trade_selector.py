"""Candidate-selection diagnostics extension."""

from __future__ import annotations

from typing import Any

from nifty_scalper_bot.strategies import trade_selector as _legacy


_LegacyTradeCandidateSelector = _legacy.TradeCandidateSelector


class HardenedTradeCandidateSelector(_LegacyTradeCandidateSelector):
    """Expose the exact configured entry-window blocker to the runner."""

    _last_entry_window_reason: str | None = None

    def select_ranked_candidates(
        self,
        *,
        direction_bias: str,
        atm_strike: int,
        snapshots: list[dict[str, Any]],
    ) -> list[Any]:
        blocked, reason = _legacy.expiry_theta_block()
        if not blocked:
            blocked, reason = _legacy.midday_pause_block()
        self._last_entry_window_reason = str(reason) if blocked and reason else None
        ranked = super().select_ranked_candidates(
            direction_bias=direction_bias,
            atm_strike=atm_strike,
            snapshots=snapshots,
        )
        if blocked:
            total = len(snapshots)
            self._last_rejects = {
                "entry_window_blocked": total,
                f"entry_window_blocked:{self._last_entry_window_reason}": total,
            }
        return ranked

    @property
    def last_entry_window_reason(self) -> str | None:
        return self._last_entry_window_reason


__all__ = ["HardenedTradeCandidateSelector"]
