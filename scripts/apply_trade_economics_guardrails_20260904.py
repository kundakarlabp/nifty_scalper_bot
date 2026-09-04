from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one occurrence, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Canonical live risk envelope: 0.75% per trade, 2% daily loss.
replace_once(
    "src/nifty_scalper_bot/config/env_utils.py",
    '''LIVE_PER_TRADE_RISK_PCT = "7.0"\n# Each live entry is capped to the remaining daily-loss budget before broker\n# submission. Keep that budget coherent with the canonical LIVE per-trade risk\n# so an indivisible NIFTY lot is not constrained by contradictory percentages.\nLIVE_DAILY_LOSS_PCT = LIVE_PER_TRADE_RISK_PCT''',
    '''LIVE_PER_TRADE_RISK_PCT = "0.75"\n# Daily loss is an independent portfolio-level circuit breaker. It must not be\n# widened merely because a single NIFTY lot is indivisible; an unaffordable\n# technical stop is skipped rather than increasing the account risk budget.\nLIVE_DAILY_LOSS_PCT = "2.0"''',
)
replace_once(
    "src/nifty_scalper_bot/config/env_utils.py",
    '''        # One canonical live risk envelope. Keep accepted aliases aligned so\n        # legacy deployment values cannot silently make the 7% per-trade policy\n        # unattainable behind a lower daily-loss ceiling. Existing remaining-day\n        # sizing clamps and final RiskManager breakers remain unchanged.''',
    '''        # One conservative canonical live risk envelope. Accepted aliases are\n        # synchronized so stale production env values cannot silently widen risk.\n        # An indivisible lot that exceeds the per-trade budget is skipped; the\n        # daily-loss circuit breaker remains an independent portfolio-level cap.''',
)

# 2) RiskSettings default + a LIVE-only hard ceiling so stale private env cannot
# silently restore the old 7% policy. Non-live research/backtests remain configurable.
replace_once(
    "src/nifty_scalper_bot/config/settings.py",
    "    per_trade_risk_pct: float = 10.0",
    "    per_trade_risk_pct: float = 0.75",
)
settings = Path("src/nifty_scalper_bot/config/settings.py")
text = settings.read_text(encoding="utf-8")
marker = "\ndef _build_risk_settings() -> RiskSettings:\n"
if text.count(marker) != 1:
    raise SystemExit("settings.py: _build_risk_settings marker mismatch")
helper = '''\ndef _real_live_risk_mode() -> bool:\n    mode = str(os.getenv("EXECUTION_MODE", "") or "").strip().upper()\n    live = str(\n        os.getenv("ENABLE_LIVE", "") or os.getenv("ENABLE_LIVE_TRADING", "")\n    ).strip().lower() in {"1", "true", "yes", "y", "on"}\n    non_live = any(\n        str(os.getenv(name, "") or "").strip().lower()\n        in {"1", "true", "yes", "y", "on"}\n        for name in ("PAPER_MODE", "PAPER__ENABLED", "SHADOW_MODE")\n    )\n    return mode == "LIVE" and live and not non_live\n\n\ndef _resolved_per_trade_risk_pct() -> float:\n    configured = _env_float(\n        "RISK__PER_TRADE_RISK_PCT",\n        "RISK_PER_TRADE_PCT",\n        default=0.75,\n        minimum=0.0,\n    )\n    if not _real_live_risk_mode():\n        return configured\n    hard_cap = _env_float(\n        "RISK_PER_TRADE_HARD_CAP_PCT", default=0.75, minimum=0.01\n    )\n    if configured > hard_cap:\n        LOGGER.warning(\n            "RISK_PER_TRADE_HARD_CAP_APPLIED configured_pct=%.4f hard_cap_pct=%.4f",\n            configured,\n            hard_cap,\n            extra={\n                "event": "RISK_PER_TRADE_HARD_CAP_APPLIED",\n                "configured_pct": configured,\n                "hard_cap_pct": hard_cap,\n            },\n        )\n    return min(configured, hard_cap)\n'''
text = text.replace(marker, helper + marker, 1)
old_builder = '''        per_trade_risk_pct=_env_float(\n            "RISK__PER_TRADE_RISK_PCT",\n            "RISK_PER_TRADE_PCT",\n            default=5.0,\n            minimum=0.0,\n        ),'''
if text.count(old_builder) != 1:
    raise SystemExit("settings.py: per_trade builder block mismatch")
text = text.replace(old_builder, "        per_trade_risk_pct=_resolved_per_trade_risk_pct(),", 1)
settings.write_text(text, encoding="utf-8")

# 3) Prospective max-position limit: at 1/1, a new risk-increasing entry is blocked.
for path in (
    "src/nifty_scalper_bot/risk/entry_guard_patch.py",
    "src/nifty_scalper_bot/risk/risk_manager.py",
):
    replace_once(path, "open_positions > max_open", "open_positions >= max_open")

# 4) Same-symbol concurrent submission race. Keep the existing timestamp map for
# orphan/recovery ownership, add lifecycle owner identity so the same recovery may
# retry while a distinct signal for the same NIFTY contract cannot race the broker.
core = Path("src/nifty_scalper_bot/execution/order_manager_core.py")
text = core.read_text(encoding="utf-8")
old_init = "        self._entries_in_flight: dict[str, float] = {}"
new_init = '''        self._entries_in_flight: dict[str, float] = {}\n        self._entry_inflight_owners: dict[str, str] = {}'''
if text.count(old_init) != 1:
    raise SystemExit("order_manager_core.py: entries_in_flight init mismatch")
text = text.replace(old_init, new_init, 1)
old_gate = '''                for _sym, _ts in list(self._entries_in_flight.items()):\n                    if _gate_now - _ts > self.ENTRY_INFLIGHT_TTL_SEC:\n                        self._entries_in_flight.pop(_sym, None)\n                conflict: str | None = None\n                for _sym in self._entries_in_flight:\n                    if _sym != normalized_symbol:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break'''
new_gate = '''                _entry_owner = str(\n                    trade_lifecycle_id or client_order_id or signal_id or trace_id or ""\n                ).strip()\n                for _sym, _ts in list(self._entries_in_flight.items()):\n                    if _gate_now - _ts > self.ENTRY_INFLIGHT_TTL_SEC:\n                        self._entries_in_flight.pop(_sym, None)\n                        self._entry_inflight_owners.pop(_sym, None)\n                conflict: str | None = None\n                for _sym in self._entries_in_flight:\n                    if _sym != normalized_symbol:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break\n                    existing_owner = self._entry_inflight_owners.get(_sym, "")\n                    if not _entry_owner or not existing_owner or existing_owner != _entry_owner:\n                        conflict = f"entry_in_flight:{_sym}"\n                        break'''
if text.count(old_gate) != 1:
    raise SystemExit(f"order_manager_core.py: atomic gate mismatch count={text.count(old_gate)}")
text = text.replace(old_gate, new_gate, 1)
old_reserve = "                self._entries_in_flight[normalized_symbol] = _gate_now"
new_reserve = '''                self._entries_in_flight[normalized_symbol] = _gate_now\n                if _entry_owner:\n                    self._entry_inflight_owners[normalized_symbol] = _entry_owner\n                else:\n                    self._entry_inflight_owners.pop(normalized_symbol, None)'''
if text.count(old_reserve) != 1:
    raise SystemExit("order_manager_core.py: reservation assignment mismatch")
text = text.replace(old_reserve, new_reserve, 1)
old_release = '''                    with self._lock:\n                        self._entries_in_flight.pop(normalized_symbol, None)'''
new_release = '''                    with self._lock:\n                        self._entries_in_flight.pop(normalized_symbol, None)\n                        self._entry_inflight_owners.pop(normalized_symbol, None)'''
if text.count(old_release) != 1:
    raise SystemExit(f"order_manager_core.py: local reservation release mismatch count={text.count(old_release)}")
text = text.replace(old_release, new_release, 1)
core.write_text(text, encoding="utf-8")

entry_geometry = Path("src/nifty_scalper_bot/execution/entry_geometry.py")
text = entry_geometry.read_text(encoding="utf-8")
old_release_geometry = '''    if lock is None:\n        reservations.pop(symbol, None)\n    else:\n        with lock:\n            reservations.pop(symbol, None)\n    return True'''
new_release_geometry = '''    owners = getattr(manager, "_entry_inflight_owners", None)\n\n    def _release() -> None:\n        reservations.pop(symbol, None)\n        if isinstance(owners, dict):\n            owners.pop(symbol, None)\n\n    if lock is None:\n        _release()\n    else:\n        with lock:\n            _release()\n    return True'''
if text.count(old_release_geometry) != 1:
    raise SystemExit("entry_geometry.py: reservation release mismatch")
entry_geometry.write_text(text.replace(old_release_geometry, new_release_geometry, 1), encoding="utf-8")

# 5) Fail-closed live direction flips at the authoritative context snapshot.
manager = Path("src/nifty_scalper_bot/core/strategy_manager.py")
text = manager.read_text(encoding="utf-8")
import_marker = "from nifty_scalper_bot.core.strategy_context_contract import"
idx = text.find(import_marker)
if idx < 0:
    raise SystemExit("strategy_manager.py: context-contract import marker missing")
line_start = text.rfind("\n", 0, idx) + 1
insert = "from nifty_scalper_bot.core.direction_stability import DirectionStabilityGate\n"
if insert not in text:
    text = text[:line_start] + insert + text[line_start:]
old_direction = '''        derived_direction, derived_confidence, derived_reasons = self._derive_context_direction(\n            direction_inputs,\n            role=role,\n        )\n        snapshot = {'''
new_direction = '''        derived_direction, derived_confidence, derived_reasons = self._derive_context_direction(\n            direction_inputs,\n            role=role,\n        )\n        direction_transition = None\n        if self._is_live_mode():\n            stability_gate = getattr(self, "_direction_stability_gate", None)\n            if stability_gate is None:\n                stability_gate = DirectionStabilityGate.from_env()\n                self._direction_stability_gate = stability_gate\n            stability = stability_gate.observe(\n                role, derived_direction, derived_confidence, now=time.time()\n            )\n            direction_transition = {\n                "pending": stability.pending,\n                "candidate_direction": stability.candidate_direction,\n                "candidate_updates": stability.candidate_updates,\n                "candidate_age_seconds": round(stability.candidate_age_seconds, 3),\n            }\n            if stability.pending:\n                derived_reasons = [\n                    *list(derived_reasons or []),\n                    "direction_transition_unconfirmed",\n                ]\n                log_throttled(\n                    log,\n                    f"direction_transition_unconfirmed:{role}",\n                    "DIRECTION_TRANSITION_UNCONFIRMED role=%s candidate=%s updates=%s age_s=%.3f",\n                    role,\n                    stability.candidate_direction,\n                    stability.candidate_updates,\n                    stability.candidate_age_seconds,\n                    interval_sec=1.0,\n                    level=logging.INFO,\n                    extra={\n                        "event": "DIRECTION_TRANSITION_UNCONFIRMED",\n                        "role": role,\n                        **direction_transition,\n                    },\n                )\n            derived_direction = stability.direction\n            derived_confidence = stability.confidence\n        snapshot = {'''
if text.count(old_direction) != 1:
    raise SystemExit(f"strategy_manager.py: direction derivation mismatch count={text.count(old_direction)}")
text = text.replace(old_direction, new_direction, 1)
old_snapshot_tail = '''            "direction_context_reasons": derived_reasons,\n            "context_timestamp_epoch": time.time(),'''
new_snapshot_tail = '''            "direction_context_reasons": derived_reasons,\n            "direction_transition": direction_transition,\n            "context_timestamp_epoch": time.time(),'''
if text.count(old_snapshot_tail) != 1:
    raise SystemExit("strategy_manager.py: direction snapshot tail mismatch")
text = text.replace(old_snapshot_tail, new_snapshot_tail, 1)
manager.write_text(text, encoding="utf-8")

# Document safety controls in the production template.
env_example = Path(".env.example")
text = env_example.read_text(encoding="utf-8")
old_risk = "RISK_PER_TRADE_PCT=0.75\n"
new_risk = "RISK_PER_TRADE_PCT=0.75\nRISK_PER_TRADE_HARD_CAP_PCT=0.75\n"
if text.count(old_risk) != 1:
    raise SystemExit(".env.example: per-trade risk marker mismatch")
text = text.replace(old_risk, new_risk, 1)
old_direction_env = "DIRECTION_LOCK_SECONDS=60\n"
new_direction_env = '''DIRECTION_LOCK_SECONDS=60\nDIRECTION_FLIP_CONFIRM_SECONDS=5.0\nDIRECTION_FLIP_CONFIRM_UPDATES=3\nDIRECTION_FLIP_MIN_CONFIDENCE=0.60\n'''
if text.count(old_direction_env) != 1:
    raise SystemExit(".env.example: direction lock marker mismatch")
env_example.write_text(text.replace(old_direction_env, new_direction_env, 1), encoding="utf-8")

print("trade economics guardrails applied")
