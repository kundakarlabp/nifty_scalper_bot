"""Purpose: Own canonical names, context roles and signal-family semantics.

Responsibilities: Normalize strategy identity; classify RangeVWAPReclaim as a
trigger only under the explicit LIVE pilot and otherwise as research context.
strategy_regime_eligible maps structural entry triggers to the canonical market
regime without replacing strategy-owned setup checks or OrderFlow context.
Safety: Canonical role mapping is authoritative across builder and manager.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from nifty_scalper_bot.config.regime_ontology import MarketRegime, normalize_regime


@dataclass(frozen=True, slots=True)
class StrategyTaxon:
    key: str
    role: str
    signal_family: str


_ALIASES: dict[str, str] = {
    "smc": "smc_lite",
    "smc_liquidity": "smc_lite",
    "smc_lite": "smc_lite",
    "smc_liquidity_sweep_lite": "smc_lite",
    "vwappro": "vwap_pro",
    "vwap_pro": "vwap_pro",
    "vwap": "vwap_pro",
    "range_vwap": "range_vwap_reclaim",
    "range_vwap_reclaim": "range_vwap_reclaim",
    "rangevwapreclaim": "range_vwap_reclaim",
    "orbpro": "orb_pro",
    "orb_pro": "orb_pro",
    "orb": "orb_pro",
    "orderflow": "order_flow",
    "order_flow": "order_flow",
    "oimaxpain": "oi_max_pain",
    "oi_max_pain": "oi_max_pain",
    "bbsqueeze": "bb_squeeze",
    "bb_squeeze": "bb_squeeze",
    "cpr": "cpr_breakout",
    "cprbreakout": "cpr_breakout",
    "cpr_breakout": "cpr_breakout",
    "rsidivergence": "rsi_divergence",
    "rsi_div": "rsi_divergence",
    "rsi_divergence": "rsi_divergence",
    "gammascalping": "gamma_scalping",
    "gamma_scalping": "gamma_scalping",
    "elite_tuesday_gamma_buyer": "tuesday_gamma_buyer",
    "tuesday_gamma_buyer": "tuesday_gamma_buyer",
}

_TAXONOMY: dict[str, StrategyTaxon] = {
    "smc_lite": StrategyTaxon("smc_lite", "trigger", "reclaim_structure"),
    "vwap_pro": StrategyTaxon("vwap_pro", "trigger", "reclaim_structure"),
    "range_vwap_reclaim": StrategyTaxon(
        "range_vwap_reclaim", "context", "range_reversion_research"
    ),
    "orb_pro": StrategyTaxon("orb_pro", "trigger", "opening_breakout"),
    "gamma_scalping": StrategyTaxon("gamma_scalping", "trigger", "expiry_trigger"),
    "tuesday_gamma_buyer": StrategyTaxon(
        "tuesday_gamma_buyer", "trigger", "expiry_trigger"
    ),
    "order_flow": StrategyTaxon("order_flow", "context", "directional_context"),
    "oi_max_pain": StrategyTaxon("oi_max_pain", "context", "directional_context"),
    "bb_squeeze": StrategyTaxon("bb_squeeze", "context", "directional_context"),
    "cpr_breakout": StrategyTaxon("cpr_breakout", "context", "directional_context"),
    "rsi_divergence": StrategyTaxon("rsi_divergence", "context", "directional_context"),
}


def normalize_strategy_name(name: str | None) -> str:
    raw = str(name or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(raw, raw)


def strategy_taxon(name: str | None) -> StrategyTaxon | None:
    return _TAXONOMY.get(normalize_strategy_name(name))


def range_vwap_live_enabled() -> bool:
    """Explicit real-broker pilot; paper, shadow and ambiguous modes fail closed."""

    def enabled(key: str) -> bool:
        return str(os.getenv(key, "false")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    return (
        str(os.getenv("EXECUTION_MODE", "SHADOW")).strip().upper() == "LIVE"
        and (enabled("ENABLE_LIVE") or enabled("ENABLE_LIVE_TRADING"))
        and enabled("ENABLE_RANGE_VWAP_LIVE")
        and not enabled("PAPER__ENABLED")
        and not enabled("PAPER_MODE")
        and not enabled("SHADOW_MODE")
    )


def canonical_strategy_role(name: str | None, *, default: str = "trigger") -> str:
    taxon = strategy_taxon(name)
    if taxon is not None and taxon.key == "range_vwap_reclaim":
        return "trigger" if range_vwap_live_enabled() else "context"
    return taxon.role if taxon is not None else default


def canonical_signal_family(
    name: str | None, *, default: str = "directional_trigger"
) -> str:
    taxon = strategy_taxon(name)
    return taxon.signal_family if taxon is not None else default


def strategy_regime_eligible(name: str | None, regime: object) -> bool:
    """Route structural triggers; strategy-owned setup gates remain binding."""
    key = normalize_strategy_name(name)
    state = normalize_regime(regime)
    eligible = {
        "range_vwap_reclaim": {MarketRegime.RANGE},
        "vwap_pro": {MarketRegime.TREND},
        "smc_lite": {
            MarketRegime.TREND, MarketRegime.RANGE,
            MarketRegime.VOLATILE, MarketRegime.EVENT,
        },
        "orb_pro": {
            MarketRegime.TREND, MarketRegime.RANGE,
            MarketRegime.VOLATILE, MarketRegime.EVENT,
        },
    }
    return state in eligible[key] if key in eligible else True


def is_context_only_strategy(name: str | None) -> bool:
    return canonical_strategy_role(name) == "context"


__all__ = [
    "StrategyTaxon",
    "canonical_signal_family",
    "canonical_strategy_role",
    "is_context_only_strategy",
    "normalize_strategy_name",
    "range_vwap_live_enabled",
    "strategy_taxon",
    "strategy_regime_eligible",
]
