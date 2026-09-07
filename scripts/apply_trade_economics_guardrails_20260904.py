from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{path}: expected one occurrence, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Canonical risk policy. LIVE normalization is the single owner of live
# aliases; do not add a second hard-cap state machine in settings/risk code.
replace_once(
    "src/nifty_scalper_bot/config/env_utils.py",
    '''LIVE_PER_TRADE_RISK_PCT = "7.0"\n# Each live entry is capped to the remaining daily-loss budget before broker\n# submission. Keep that budget coherent with the canonical LIVE per-trade risk\n# so an indivisible NIFTY lot is not constrained by contradictory percentages.\nLIVE_DAILY_LOSS_PCT = LIVE_PER_TRADE_RISK_PCT''',
    '''LIVE_PER_TRADE_RISK_PCT = "0.75"\n# Daily loss is an independent portfolio-level circuit breaker. An indivisible\n# lot that cannot fit the per-trade risk budget is skipped rather than widening\n# either limit at runtime.\nLIVE_DAILY_LOSS_PCT = "2.0"''',
)
replace_once(
    "src/nifty_scalper_bot/config/env_utils.py",
    '''        # One canonical live risk envelope. Keep accepted aliases aligned so\n        # legacy deployment values cannot silently make the 7% per-trade policy\n        # unattainable behind a lower daily-loss ceiling. Existing remaining-day\n        # sizing clamps and final RiskManager breakers remain unchanged.''',
    '''        # One conservative canonical live risk envelope. Accepted aliases are\n        # synchronized here so stale deployment values cannot silently widen\n        # risk. Existing sizing clamps and final RiskManager breakers remain\n        # unchanged.''',
)
replace_once(
    "src/nifty_scalper_bot/config/settings.py",
    "    per_trade_risk_pct: float = 10.0",
    "    per_trade_risk_pct: float = 0.75",
)
replace_once(
    "src/nifty_scalper_bot/config/settings.py",
    '''        per_trade_risk_pct=_env_float(\n            "RISK__PER_TRADE_RISK_PCT",\n            "RISK_PER_TRADE_PCT",\n            default=5.0,\n            minimum=0.0,\n        ),''',
    '''        per_trade_risk_pct=_env_float(\n            "RISK__PER_TRADE_RISK_PCT",\n            "RISK_PER_TRADE_PCT",\n            default=0.75,\n            minimum=0.0,\n        ),''',
)

# 2) Prospective single-position limit. can_trade/_daily_limit_block_reason are
# entry gates, so 1/1 must reject the next risk-increasing entry.
replace_once(
    "src/nifty_scalper_bot/risk/entry_guard_patch.py",
    "if open_positions > max_open:",
    "if open_positions >= max_open:",
)
replace_once(
    "src/nifty_scalper_bot/risk/risk_manager.py",
    "if max_open > 0 and open_positions > max_open:",
    "if max_open > 0 and open_positions >= max_open:",
)

# 3) Keep the existing atomic entry reservation architecture, but attach an
# owner identity. A distinct same-symbol signal cannot race the broker; the
# entry-recovery wrapper may retry the exact same lifecycle while it owns the
# reservation.
core = Path("src/nifty_scalper_bot/execution/order_manager_core.py")
text = core.read_text(encoding="utf-8")
old_init = '''        self._entries_in_flight: dict[str, float] = {}'''
new_init = '''        self._entries_in_flight: dict[str, float] = {}\n        self._entry_inflight_owners: dict[str, str] = {}'''
if text.count(old_init) != 1:
    raise SystemExit("order_manager_core.py: entries-in-flight init mismatch")
text = text.replace(old_init, new_init, 1)

old_gate = '''                # Purge expired reservations first (self-healing).\n                for _sym, _ts in list(self._entries_in_flight.items()):\n                    if _gate_now - _ts > self.ENTRY_INFLIGHT_TTL_SEC:\n                        self._entries_in_flight.pop(_sym, None)\n                conflict: str | None = None\n                for _sym in self._entries_in_flight:\n                    if _sym != normalized_symbol:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break'''
new_gate = '''                # Purge expired reservations first (self-healing).\n                for _sym, _ts in list(self._entries_in_flight.items()):\n                    if _gate_now - _ts > self.ENTRY_INFLIGHT_TTL_SEC:\n                        self._entries_in_flight.pop(_sym, None)\n                        self._entry_inflight_owners.pop(_sym, None)\n                _entry_owner = str(\n                    trade_lifecycle_id or client_order_id or signal_id or ""\n                ).strip()\n                conflict: str | None = None\n                for _sym in self._entries_in_flight:\n                    if _sym != normalized_symbol:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break\n                    existing_owner = self._entry_inflight_owners.get(_sym, "")\n                    same_recovery = bool(\n                        getattr(self, "_entry_recovery_active", False)\n                        and _entry_owner\n                        and existing_owner == _entry_owner\n                    )\n                    if not same_recovery:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break'''
if text.count(old_gate) != 1:
    raise SystemExit(f"order_manager_core.py: entry gate mismatch={text.count(old_gate)}")
text = text.replace(old_gate, new_gate, 1)

old_reserve = '''                self._entries_in_flight[normalized_symbol] = _gate_now'''
new_reserve = '''                self._entries_in_flight[normalized_symbol] = _gate_now\n                if _entry_owner:\n                    self._entry_inflight_owners[normalized_symbol] = _entry_owner\n                else:\n                    self._entry_inflight_owners.pop(normalized_symbol, None)'''
if text.count(old_reserve) != 1:
    raise SystemExit("order_manager_core.py: reservation assignment mismatch")
text = text.replace(old_reserve, new_reserve, 1)

old_release = '''                    with self._lock:\n                        self._entries_in_flight.pop(normalized_symbol, None)'''
new_release = '''                    with self._lock:\n                        self._entries_in_flight.pop(normalized_symbol, None)\n                        self._entry_inflight_owners.pop(normalized_symbol, None)'''
if text.count(old_release) != 1:
    raise SystemExit(
        f"order_manager_core.py: local reservation release mismatch={text.count(old_release)}"
    )
text = text.replace(old_release, new_release, 1)
core.write_text(text, encoding="utf-8")

entry_geometry = Path("src/nifty_scalper_bot/execution/entry_geometry.py")
text = entry_geometry.read_text(encoding="utf-8")
old_release_geometry = '''    if lock is None:\n        reservations.pop(symbol, None)\n    else:\n        with lock:\n            reservations.pop(symbol, None)\n    return True'''
new_release_geometry = '''    owners = getattr(manager, "_entry_inflight_owners", None)\n\n    def _release() -> None:\n        reservations.pop(symbol, None)\n        if isinstance(owners, dict):\n            owners.pop(symbol, None)\n\n    if lock is None:\n        _release()\n    else:\n        with lock:\n            _release()\n    return True'''
if text.count(old_release_geometry) != 1:
    raise SystemExit("entry_geometry.py: reservation release mismatch")
entry_geometry.write_text(
    text.replace(old_release_geometry, new_release_geometry, 1), encoding="utf-8"
)

# 4) Direction persistence stays in its existing owner, StrategyOrchestrator.
# Retain the accepted-entry direction after exit for DIRECTION_LOCK_SECONDS;
# opposite re-entry is blocked during that window, same-direction is not.
orch = Path("src/nifty_scalper_bot/strategies/orchestrator.py")
text = orch.read_text(encoding="utf-8")
old_stale = '''            if self._active_direction and not self._has_open_position_for_locked_symbol(position_manager):\n                self.clear_direction_lock(reason="stale_lock_no_open_position", symbol=symbol)\n\n            if _direction and self._active_direction:'''
new_stale = '''            if self._active_direction and not self._has_open_position_for_locked_symbol(position_manager):\n                _time_since_lock = _t.time() - self._direction_lock_time\n                if _time_since_lock >= _dir_cooldown:\n                    self.clear_direction_lock(\n                        reason="direction_lock_expired", symbol=symbol\n                    )\n\n            if _direction and self._active_direction:'''
if text.count(old_stale) != 1:
    raise SystemExit("orchestrator.py: stale direction-lock block mismatch")
text = text.replace(old_stale, new_stale, 1)

old_reconcile = '''        if self._has_open_position_for_locked_symbol(position_manager):\n            return\n        self.clear_direction_lock(reason="resolved_bias_flip", symbol=symbol)'''
new_reconcile = '''        if self._has_open_position_for_locked_symbol(position_manager):\n            return\n        import time as _t\n\n        cooldown = parse_float_env(os.getenv("DIRECTION_LOCK_SECONDS"), 10.0)\n        if _t.time() - self._direction_lock_time < cooldown:\n            return\n        self.clear_direction_lock(reason="resolved_bias_flip", symbol=symbol)'''
if text.count(old_reconcile) != 1:
    raise SystemExit("orchestrator.py: reconcile direction block mismatch")
text = text.replace(old_reconcile, new_reconcile, 1)

old_exit = '''        with self._lock:\n            self._active.pop(normalized, None)\n            self._pending_underlyings.pop(normalized, None)\n            self.clear_direction_lock(reason="notify_exit", symbol=normalized)'''
new_exit = '''        with self._lock:\n            self._active.pop(normalized, None)\n            self._pending_underlyings.pop(normalized, None)\n        if self._active_direction:\n            self._logger.info(\n                "DIRECTION_LOCK_RETAINED_AFTER_EXIT direction=%s symbol=%s",\n                self._active_direction,\n                self._active_direction_symbol,\n                extra={\n                    "event": "orchestrator_direction_lock_retained_after_exit",\n                    "direction": self._active_direction,\n                    "symbol": self._active_direction_symbol,\n                },\n            )'''
if text.count(old_exit) != 1:
    raise SystemExit("orchestrator.py: notify_exit direction clear mismatch")
text = text.replace(old_exit, new_exit, 1)
orch.write_text(text, encoding="utf-8")

print("canonical trade-economics guardrails applied")
