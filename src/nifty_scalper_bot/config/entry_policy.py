"""File purpose:
    Own the option premium and spread admission thresholds used by evaluation
    and execution.

Key responsibilities:
    - Resolve evaluation vs execution floors and caps from configuration.
    - Enforce evaluation_min_premium <= execution_min_premium.
    - Enforce evaluation_max_spread_pct >= execution_max_spread_pct.

Operational constraints:
    - Evaluation may be wider than execution; execution is the binding constraint.
    - Invalid or inverted configuration is repaired towards the safer side and
      logged rather than raising mid-session.
    - Gates must name which policy they enforce; they must not invent numbers.

Single owner of the option premium and spread admission thresholds.

The same two decisions were previously expressed as unnamed numbers in three
layers: the runner evaluation pregate read ``MIN_OPTION_PREMIUM`` with a
default of 20 while ``TradeCandidateSelector`` read the *same* variable with a
default of 40, and the spread ceiling was 1.5% at evaluation, 0.75% at
candidate selection and 10% in the quality helper's fallback. Because the two
premium floors shared one environment variable, setting it collapsed the
distinction the defaults were expressing.

The distinction is deliberate and is preserved here as two named policies:

* **Evaluation** decides what the strategy layer is allowed to look at.
  A wider universe is legitimate; observing a contract costs nothing.
* **Execution** decides what may actually be traded. This is the binding
  constraint and is never looser than evaluation.

The invariants are therefore ``evaluation_min_premium <= execution_min_premium``
and ``evaluation_max_spread_pct >= execution_max_spread_pct``. A configuration
that inverts either one is repaired towards the safer side (tighter execution)
rather than silently honoured, and every gate states which policy it enforces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

__all__ = ["EntryPolicy", "resolve_entry_policy"]

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}

# Defaults reproduce the thresholds the layers were already using, so this
# module changes ownership and naming rather than live admission behaviour.
_DEFAULT_EVALUATION_MIN_PREMIUM = 20.0
_DEFAULT_EXECUTION_MIN_PREMIUM = 40.0
_DEFAULT_MAX_PREMIUM = 650.0
_DEFAULT_EVALUATION_MAX_SPREAD_PCT = 1.5
_DEFAULT_EXECUTION_MAX_SPREAD_PCT_LIVE = 0.75


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        LOGGER.warning(
            "ENTRY_POLICY_INVALID_VALUE setting=%s value=%s using_default=%s",
            name,
            raw,
            default,
            extra={
                "event": "ENTRY_POLICY_INVALID_VALUE",
                "setting": name,
                "default": default,
            },
        )
        return default
    return value


def _env_true(name: str) -> bool:
    return str(os.getenv(name, "") or "").strip().lower() in _TRUE_VALUES


def _real_live_mode() -> bool:
    """Mirror the live-mode test used by the execution-side gates."""
    mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
    live_enabled = _env_true("ENABLE_LIVE") or _env_true("ENABLE_LIVE_TRADING")
    paper_or_shadow = (
        _env_true("PAPER_MODE")
        or _env_true("PAPER__ENABLED")
        or _env_true("SHADOW_MODE")
    )
    return mode == "LIVE" and live_enabled and not paper_or_shadow


@dataclass(frozen=True, slots=True)
class EntryPolicy:
    """Resolved admission thresholds for one evaluation cycle."""

    evaluation_min_premium: float
    execution_min_premium: float
    max_premium: float
    evaluation_max_spread_pct: float
    execution_max_spread_pct: float
    is_live: bool

    def describe(self) -> dict[str, float | bool]:
        """Return the policy as log/metadata fields."""
        return {
            "evaluation_min_premium": self.evaluation_min_premium,
            "execution_min_premium": self.execution_min_premium,
            "max_premium": self.max_premium,
            "evaluation_max_spread_pct": self.evaluation_max_spread_pct,
            "execution_max_spread_pct": self.execution_max_spread_pct,
            "policy_live": self.is_live,
        }


def resolve_entry_policy() -> EntryPolicy:
    """Return the current entry policy with its invariants enforced.

    Returns:
        An :class:`EntryPolicy` whose evaluation bounds are never stricter
        than its execution bounds.

    Raises:
        None. Invalid or inverted configuration is repaired towards the safer
        side and logged rather than raised, so a misconfiguration cannot take
        the runtime down mid-session.
    """
    is_live = _real_live_mode()

    # MIN_OPTION_PREMIUM stays honoured for both policies so existing operator
    # configuration keeps its current meaning; the split defaults only apply
    # when nothing is configured.
    shared_min = _env_float("MIN_OPTION_PREMIUM", None)
    evaluation_min = _env_float(
        "EVALUATION_MIN_OPTION_PREMIUM",
        shared_min if shared_min is not None else _DEFAULT_EVALUATION_MIN_PREMIUM,
    )
    execution_min = _env_float(
        "EXECUTION_MIN_OPTION_PREMIUM",
        shared_min if shared_min is not None else _DEFAULT_EXECUTION_MIN_PREMIUM,
    )
    max_premium = _env_float("MAX_OPTION_PREMIUM", _DEFAULT_MAX_PREMIUM)

    evaluation_spread = _env_float(
        "MAX_OPTION_SPREAD_PCT_FOR_EVAL", _DEFAULT_EVALUATION_MAX_SPREAD_PCT
    )
    # One execution-spread owner. The explicit canonical/live-candidate
    # settings retain their historical ability to configure the binding cap.
    # Older quality/order aliases are accepted only as *tighter* fallbacks so
    # they can never silently loosen the established 0.75% live default.
    canonical_execution_spread = _env_float("EXECUTION_MAX_OPTION_SPREAD_PCT", None)
    live_candidate_spread = _env_float("LIVE_CANDIDATE_MAX_SPREAD_PCT", None)
    if canonical_execution_spread is not None:
        execution_spread = canonical_execution_spread
    elif live_candidate_spread is not None:
        execution_spread = live_candidate_spread
    else:
        legacy_caps = [
            value
            for value in (
                _env_float("ORDER_MAX_SPREAD_PCT", None),
                _env_float("SPREAD_MAX_PCT", None),
            )
            if value is not None and value > 0
        ]
        execution_spread = min(
            [_DEFAULT_EXECUTION_MAX_SPREAD_PCT_LIVE, *legacy_caps]
        )

    evaluation_min = max(0.0, float(evaluation_min or 0.0))
    execution_min = max(0.0, float(execution_min or 0.0))
    max_premium = max(0.0, float(max_premium or 0.0))
    evaluation_spread = max(0.01, float(evaluation_spread or 0.01))
    execution_spread = max(0.01, float(execution_spread or 0.01))

    if evaluation_min > execution_min:
        LOGGER.warning(
            "ENTRY_POLICY_INVARIANT_REPAIRED field=min_premium "
            "evaluation=%s execution=%s resolution=raise_execution_floor",
            evaluation_min,
            execution_min,
            extra={
                "event": "ENTRY_POLICY_INVARIANT_REPAIRED",
                "field": "min_premium",
                "evaluation": evaluation_min,
                "execution": execution_min,
            },
        )
        execution_min = evaluation_min

    if evaluation_spread < execution_spread:
        LOGGER.warning(
            "ENTRY_POLICY_INVARIANT_REPAIRED field=max_spread_pct "
            "evaluation=%s execution=%s resolution=tighten_execution_cap",
            evaluation_spread,
            execution_spread,
            extra={
                "event": "ENTRY_POLICY_INVARIANT_REPAIRED",
                "field": "max_spread_pct",
                "evaluation": evaluation_spread,
                "execution": execution_spread,
            },
        )
        execution_spread = evaluation_spread

    return EntryPolicy(
        evaluation_min_premium=evaluation_min,
        execution_min_premium=execution_min,
        max_premium=max_premium,
        evaluation_max_spread_pct=evaluation_spread,
        execution_max_spread_pct=execution_spread,
        is_live=is_live,
    )
