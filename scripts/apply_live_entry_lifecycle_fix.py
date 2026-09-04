from __future__ import annotations

from pathlib import Path


RUNTIME = Path("src/nifty_scalper_bot/execution/runtime_order_manager.py")
RUNNER = Path("src/nifty_scalper_bot/strategies/runner.py")
TESTS = Path("tests/execution/test_live_entry_lifecycle_regressions.py")


def replace_exact(path: Path, old: str, new: str, *, expected: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} exact matches, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


replace_exact(
    RUNTIME,
    '''    def current_entry_blocker(self) -> Mapping[str, Any] | None:\n        return _current_entry_blocker(self)\n''',
    '''    def _release_resolved_entry_reconciliation_blocker(\n        self, blocker: Mapping[str, Any]\n    ) -> Mapping[str, Any] | None:\n        """Release only a terminal-unfilled entry blocker proven broker-flat.\n\n        The entry-recovery latch is intentionally fail-closed while broker truth is\n        uncertain. Once the same broker order is authoritatively terminal-unfilled\n        and the canonical bracket authority proves zero broker exposure for the same\n        symbol, keeping the manager-global latch would block unrelated future entries\n        indefinitely. Broker I/O occurs outside the OrderManager lock; the lock is\n        used only for the final identity-checked state transition so a newer blocker\n        cannot be cleared by an older reconciliation result.\n        """\n        if str(blocker.get("block_reason") or "").strip().lower() != (\n            "entry_reconciliation_pending"\n        ):\n            return blocker\n        details = blocker.get("details")\n        if not isinstance(details, Mapping):\n            return blocker\n        order_id = str(details.get("order_id") or "").strip()\n        symbol = str(details.get("symbol") or "").strip()\n        if not order_id or not symbol:\n            return blocker\n\n        authority = getattr(self, "_bracket_manager", None)\n        order_status = getattr(authority, "_broker_entry_order_status", None)\n        broker_quantity = getattr(authority, "_broker_position_quantity", None)\n        if not callable(order_status) or not callable(broker_quantity):\n            return blocker\n\n        try:\n            status_payload, status_known = order_status(order_id)\n        except Exception:\n            return blocker\n        if not status_known or not isinstance(status_payload, Mapping):\n            return blocker\n        status = str(status_payload.get("status") or "").strip().upper()\n        if status not in {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}:\n            return blocker\n\n        try:\n            quantity = broker_quantity(symbol)\n        except Exception:\n            return blocker\n        if quantity is None:\n            return blocker\n        try:\n            if int(quantity) != 0:\n                return blocker\n        except (TypeError, ValueError):\n            return blocker\n\n        def _clear_if_current() -> Mapping[str, Any] | None:\n            current = getattr(self, "_entry_lifecycle_blocker", None)\n            if current is not blocker:\n                return current if isinstance(current, Mapping) else None\n            self._entry_lifecycle_blocker = None\n            if getattr(self, "_last_order_decision", None) is blocker:\n                self._last_order_decision = {}\n            return None\n\n        lock = getattr(self, "_lock", None)\n        if lock is None:\n            remaining = _clear_if_current()\n        else:\n            try:\n                with lock:\n                    remaining = _clear_if_current()\n            except Exception:\n                return blocker\n\n        if remaining is None:\n            logger = getattr(self, "_logger", None)\n            log = getattr(logger, "info", None)\n            if callable(log):\n                log(\n                    "ENTRY_RECONCILIATION_RESOLVED_FLAT order_id=%s symbol=%s status=%s",\n                    order_id,\n                    symbol,\n                    status,\n                    extra={\n                        "event": "ENTRY_RECONCILIATION_RESOLVED_FLAT",\n                        "order_id": order_id,\n                        "symbol": symbol,\n                        "order_status": status,\n                    },\n                )\n        return remaining\n\n    def current_entry_blocker(self) -> Mapping[str, Any] | None:\n        blocker = _current_entry_blocker(self)\n        if not isinstance(blocker, Mapping):\n            return None\n        return self._release_resolved_entry_reconciliation_blocker(blocker)\n''',
)

replace_exact(
    RUNNER,
    '''            self._last_selected_candidate_eval_completed_at = (\n                self._last_entry_eval_completed_at\n            )\n''',
    '''            self._last_selected_candidate_eval_completed_ts = (\n                self._last_entry_eval_completed_at\n            )\n''',
)

replace_exact(
    TESTS,
    '''    def _position_flat_for_symbol(self, symbol: str) -> bool:\n        assert symbol == SYMBOL\n        return self.flat\n''',
    '''    def _broker_position_quantity(self, symbol: str) -> int:\n        assert symbol == SYMBOL\n        return 0 if self.flat else 65\n''',
)

replace_exact(
    TESTS,
    '''        def _position_flat_for_symbol(self, symbol: str) -> bool:\n            manager._entry_lifecycle_blocker = newer\n            return super()._position_flat_for_symbol(symbol)\n''',
    '''        def _broker_position_quantity(self, symbol: str) -> int:\n            manager._entry_lifecycle_blocker = newer\n            return super()._broker_position_quantity(symbol)\n''',
)

print("Applied canonical live-entry lifecycle corrections")
