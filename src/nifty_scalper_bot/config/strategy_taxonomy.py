"""Canonical strategy identity, role and signal-family taxonomy.

Strategy modules may emit richer metadata, but structural identity belongs here so
builder, manager and runner cannot disagree about whether evidence may trigger a
trade.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    "orbpro": "orb_pro",
    "orb_pro": "orb_pro",
    "orb": "orb_pro",
    "premium_momentum": "premium_squeeze",
    "premium_momentum_squeeze": "premium_squeeze",
    "premium_squeeze": "premium_squeeze",
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
    "orb_pro": StrategyTaxon("orb_pro", "trigger", "opening_breakout"),
    "premium_squeeze": StrategyTaxon(
        "premium_squeeze", "trigger", "volatility_momentum"
    ),
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


def canonical_strategy_role(name: str | None, *, default: str = "trigger") -> str:
    taxon = strategy_taxon(name)
    return taxon.role if taxon is not None else default


def canonical_signal_family(
    name: str | None, *, default: str = "directional_trigger"
) -> str:
    taxon = strategy_taxon(name)
    return taxon.signal_family if taxon is not None else default


def is_context_only_strategy(name: str | None) -> bool:
    return canonical_strategy_role(name) == "context"


__all__ = [
    "StrategyTaxon",
    "canonical_signal_family",
    "canonical_strategy_role",
    "is_context_only_strategy",
    "normalize_strategy_name",
    "strategy_taxon",
]
