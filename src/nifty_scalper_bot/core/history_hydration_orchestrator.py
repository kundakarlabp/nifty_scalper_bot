"""Canonical active-basket history hydration planner/executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nifty_scalper_bot.data.market_data_manager import HydrationResult
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SymbolHydrationRequirement:
    symbol: str
    role: str
    required_bars: int
    target_bars: int
    gating: bool


@dataclass(frozen=True, slots=True)
class HydrationPlan:
    basket_version: str | None
    requirements: tuple[SymbolHydrationRequirement, ...]

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (
            self.basket_version,
            tuple(
                (r.symbol, r.role, r.required_bars, r.target_bars, r.gating)
                for r in self.requirements
            ),
        )


@dataclass(frozen=True, slots=True)
class HydrationPlanResult:
    results_by_symbol: Mapping[str, HydrationResult]
    started_fetches: int
    joined_inflight: int
    failed_symbols: tuple[str, ...]
    gating_failures: tuple[str, ...]


def _basket_get(basket: Any, key: str, default: Any = None) -> Any:
    if isinstance(basket, Mapping):
        return basket.get(key, default)
    return getattr(basket, key, default)


def _canon(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def build_active_basket_hydration_plan(
    ctx: Any,
    *,
    required_option_bars: int,
    required_context_bars: int,
    target_option_bars: int | None = None,
    target_context_bars: int | None = None,
) -> HydrationPlan:
    """Build one deduplicated plan from the active basket SSOT."""
    basket = (
        getattr(ctx, "active_contract_basket", None)
        or getattr(ctx, "active_trading_universe", {})
        or {}
    )
    version = (
        str(
            _basket_get(basket, "basket_version", None)
            or _basket_get(basket, "version", None)
            or ""
        )
        or None
    )
    reqs: list[SymbolHydrationRequirement] = []
    spot = _canon(
        _basket_get(basket, "spot_symbol", None)
        or getattr(ctx, "nifty_symbol", None)
        or "NSE:NIFTY"
    )
    fut = _canon(
        _basket_get(basket, "futures_symbol", None)
        or _basket_get(basket, "future_symbol", None)
        or ""
    )
    ce = _canon(
        _basket_get(basket, "selected_ce", None)
        or _basket_get(basket, "atm_ce", None)
        or ""
    )
    pe = _canon(
        _basket_get(basket, "selected_pe", None)
        or _basket_get(basket, "atm_pe", None)
        or ""
    )
    if spot:
        reqs.append(
            SymbolHydrationRequirement(
                spot,
                "spot",
                required_context_bars,
                target_context_bars or required_context_bars,
                True,
            )
        )
    if fut:
        reqs.append(
            SymbolHydrationRequirement(
                fut,
                "futures_context",
                required_context_bars,
                target_context_bars or required_context_bars,
                True,
            )
        )
    if ce:
        reqs.append(
            SymbolHydrationRequirement(
                ce,
                "selected_ce",
                required_option_bars,
                target_option_bars or required_option_bars,
                True,
            )
        )
    if pe:
        reqs.append(
            SymbolHydrationRequirement(
                pe,
                "selected_pe",
                required_option_bars,
                target_option_bars or required_option_bars,
                True,
            )
        )
    for sym in (
        _basket_get(basket, "option_symbols", None)
        or _basket_get(basket, "symbols", ())
        or ()
    ):
        cs = _canon(sym)
        if cs and cs not in {ce, pe} and cs.endswith(("CE", "PE")):
            reqs.append(
                SymbolHydrationRequirement(
                    cs,
                    "option_context",
                    required_option_bars,
                    target_option_bars or required_option_bars,
                    False,
                )
            )
    merged: dict[str, SymbolHydrationRequirement] = {}
    for req in reqs:
        old = merged.get(req.symbol)
        if old is None or (req.required_bars, req.target_bars, req.gating) > (
            old.required_bars,
            old.target_bars,
            old.gating,
        ):
            merged[req.symbol] = req
    return HydrationPlan(version, tuple(merged.values()))


async def execute_hydration_plan(
    ctx: Any, plan: HydrationPlan, *, phase: str, reason: str
) -> HydrationPlanResult:
    """Execute one MDM ensure per canonical symbol.

    Runner/indicator propagation is performed once after a successful ensure.
    """
    mdm = getattr(ctx, "market_data_manager", None)
    runner = getattr(ctx, "strategy_runner", None)
    results: dict[str, HydrationResult] = {}
    failed: list[str] = []
    gating_failures: list[str] = []
    started = 0
    joined = 0
    ensure = getattr(mdm, "ensure_history", None)
    if not callable(ensure):
        return HydrationPlanResult(
            {},
            0,
            0,
            tuple(r.symbol for r in plan.requirements),
            tuple(r.symbol for r in plan.requirements if r.gating),
        )
    for req in plan.requirements:
        result = await ensure(
            req.symbol,
            interval="minute",
            required_bars=req.required_bars,
            target_bars=req.target_bars,
            role=req.role,
            phase=phase,
            reason=reason,
            minimum_only=False,
        )
        results[req.symbol] = result
        started += int(bool(getattr(result, "broker_fetch_started", False)))
        joined += int(bool(getattr(result, "joined_inflight", False)))
        if getattr(result, "failure_reason", None):
            failed.append(req.symbol)
            if req.gating:
                gating_failures.append(req.symbol)
            continue
        sync = getattr(runner, "sync_history_from_mdm", None)
        if callable(sync):
            sync(
                req.symbol,
                required_bars=req.required_bars,
                reason=reason,
                role=req.role,
                request_if_short=False,
            )
    LOGGER.info(
        "HYDRATION_PLAN_RESULT basket_version=%s symbol_count=%s "
        "started_fetches=%s joined_inflight=%s failed_symbols=%s "
        "gating_failures=%s phase=%s reason=%s",
        plan.basket_version,
        len(plan.requirements),
        started,
        joined,
        failed,
        gating_failures,
        phase,
        reason,
        extra={
            "event": "HYDRATION_PLAN_RESULT",
            "basket_version": plan.basket_version,
            "symbol_count": len(plan.requirements),
            "started_fetches": started,
            "joined_inflight": joined,
            "failed_symbols": failed,
            "gating_failures": gating_failures,
            "phase": phase,
            "reason": reason,
        },
    )
    return HydrationPlanResult(
        results, started, joined, tuple(failed), tuple(gating_failures)
    )
