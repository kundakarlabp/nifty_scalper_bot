from __future__ import annotations

from scripts._execution_patch_utils import assert_parses, replace_method, replace_once

PATH = "src/nifty_scalper_bot/execution/position_manager.py"

replace_once(
    PATH,
    "from nifty_scalper_bot.infra.metrics import METRICS\n",
    "from nifty_scalper_bot.infra.metrics import METRICS\n"
    "from nifty_scalper_bot.execution.position_snapshot import (\n"
    "    PositionSnapshotError,\n"
    "    decode_position_snapshot,\n"
    ")\n",
)
replace_once(
    PATH,
    "        self._daily_realized_pnl: float = 0.0\n",
    "        self._daily_realized_pnl: float = 0.0\n"
    "        self._local_realized_pnl: float = 0.0\n"
    "        self._broker_realized_pnl: float | None = None\n",
)

replace_method(
    PATH,
    "PositionManager",
    "_handle_reconcile_failure",
    '''
def _handle_reconcile_failure(
    self,
    *,
    reason: str,
    error: Exception | None,
    payload_count: int,
    previous_positions: Mapping[str, Position] | None,
) -> None:
    """Record failure without replacing newer local fill/position state."""
    self._last_reconcile_attempt = _now()
    self._consecutive_reconcile_failures += 1
    self._last_reconcile_error = str(error) if error is not None else reason
    reason_token = canonical(reason)
    event_key = f"failure:{self._last_reconcile_attempt.isoformat()}:{reason_token}"
    try:
        METRICS.record_broker_sync(
            success=False,
            reason=reason_token,
            latency_seconds=None,
            event_id=event_key,
        )
        METRICS.increment_retry_event(
            label="position_reconcile",
            stage="apply",
            outcome=reason_token,
        )
    except Exception as metrics_exc:  # noqa: BLE001
        self._logger.error(
            "Failure in reconcile failure metrics: %s",
            metrics_exc,
            extra={"event": "position_reconcile_metric_failure"},
        )
    with self._lock:
        preserved_count = len(self._positions)
    retry_delay = self._compute_retry_delay()
    payload: dict[str, object] = {
        "reason": reason_token,
        "failures": self._consecutive_reconcile_failures,
        "retry_sec": retry_delay,
        "count": payload_count,
        "timestamp": self._last_reconcile_attempt.isoformat(),
        "restored": False,
        "source": "current_state_preserved",
        "preserved_count": preserved_count,
    }
    if error is not None:
        payload["error"] = str(error)
    try:
        _POSITION_RECONCILE_EVENTS.labels("failed").inc()
    except Exception:  # noqa: BLE001
        pass
    self._notify_reconcile_event("position_reconcile_failed", payload)
    self._schedule_retry_after_failure(retry_delay)
''',
)

replace_method(
    PATH,
    "PositionManager",
    "reconcile_now",
    '''
def reconcile_now(self) -> bool:
    """Fetch and atomically apply one authoritative broker snapshot."""
    payload_count = 0
    fetcher = self._resolve_broker_position_fetcher()
    if fetcher is None:
        self._handle_reconcile_failure(
            reason=canonical("fetcher_missing"),
            error=None,
            payload_count=0,
            previous_positions=None,
        )
        return False
    try:
        response = fetcher()
        snapshot = decode_position_snapshot(response)
    except Exception as exc:  # noqa: BLE001
        reason = canonical(
            "payload_invalid" if isinstance(exc, PositionSnapshotError) else "fetch_error"
        )
        self._logger.warning(
            "Position reconciliation snapshot failed: %s",
            exc,
            extra={"event": "position_reconcile_failed", "reason": reason},
            exc_info=exc,
        )
        self._handle_reconcile_failure(
            reason=reason,
            error=exc,
            payload_count=0,
            previous_positions=None,
        )
        return False

    payloads = snapshot.raw_rows()
    payload_count = len(payloads)
    try:
        self.synchronize_with_broker(payloads)
    except Exception as exc:  # noqa: BLE001
        reason = canonical("apply_error")
        self._logger.warning(
            "Position reconciliation apply failed: %s",
            exc,
            extra={"event": "position_reconcile_failed", "reason": reason},
            exc_info=exc,
        )
        self._handle_reconcile_failure(
            reason=reason,
            error=exc,
            payload_count=payload_count,
            previous_positions=None,
        )
        return False

    self._logger.info(
        "POSITION_RECONCILE_OK count=%s source=%s",
        payload_count,
        snapshot.source,
        extra={
            "event": "position_reconcile_ok",
            "count": payload_count,
            "source": snapshot.source,
        },
    )
    self._handle_reconcile_success(payload_count)
    return True
''',
)

replace_method(
    PATH,
    "PositionManager",
    "synchronize_with_broker",
    '''
def synchronize_with_broker(
    self, broker_positions: Sequence[Mapping[str, object]]
) -> None:
    """Validate and atomically replace managed positions from broker truth."""
    snapshot = decode_position_snapshot(broker_positions)

    def get_float(
        record: Mapping[str, object],
        keys: Sequence[str],
        *,
        default: float = 0.0,
    ) -> float:
        for key in keys:
            if key not in record or record.get(key) is None:
                continue
            try:
                value = float(cast(Any, record.get(key)))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid broker numeric field {key}={record.get(key)!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"invalid broker numeric field {key}={record.get(key)!r}"
                )
            return value
        return float(default)

    with self._lock:
        existing_positions = copy.deepcopy(self._positions)
        reconciled: Dict[str, Position] = {}
        snapshot_realized_pnl = 0.0
        snapshot_realized_seen = False
        for row in snapshot.rows:
            record = row.raw
            symbol = row.symbol
            if not is_strategy_instrument(symbol):
                continue
            product = str(record.get("product") or "").strip().upper()
            if product != "MIS":
                if symbol in existing_positions:
                    raise ValueError(
                        f"managed broker position {symbol} has unexpected product "
                        f"{product or 'missing'}"
                    )
                continue
            quantity = row.quantity
            realized_pnl = get_float(
                record, ("realised", "realized"), default=0.0
            )
            if "realised" in record or "realized" in record:
                snapshot_realized_seen = True
                snapshot_realized_pnl += realized_pnl
            if quantity == 0:
                continue
            side: Side = "LONG" if quantity > 0 else "SHORT"
            abs_quantity = abs(quantity)
            entry_price = get_float(
                record,
                ("average_price", "avg_price", "price", "buy_price"),
            )
            current_price = get_float(
                record,
                ("last_price", "ltp", "close", "sell_price"),
                default=entry_price,
            )
            if entry_price <= 0.0 and current_price > 0.0:
                entry_price = current_price
            if current_price <= 0.0 and entry_price > 0.0:
                current_price = entry_price
            if entry_price <= 0.0 or current_price <= 0.0:
                raise ValueError(f"broker position {symbol} has no valid price")
            existing = existing_positions.get(symbol)
            if existing is None:
                position = self._create_position(
                    symbol=symbol,
                    quantity=abs_quantity,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    realized_pnl=realized_pnl,
                    source="broker_sync",
                )
            else:
                position = self._update_position(
                    position=existing,
                    quantity=abs_quantity,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    realized_pnl=realized_pnl,
                    source="broker_sync",
                )
            reconciled[symbol] = position

        old_keys = set(self._positions)
        new_keys = set(reconciled)
        removed_symbols = sorted(old_keys - new_keys)
        added_symbols = sorted(new_keys - old_keys)
        self._positions = reconciled
        if snapshot_realized_seen:
            self._broker_realized_pnl = float(snapshot_realized_pnl)
            self._refresh_realized_pnl_locked()

    if removed_symbols:
        hook = getattr(self, "_on_symbols_flat_hook", None)
        if hook is not None:
            try:
                hook(list(removed_symbols))
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in on_symbols_flat hook: %s",
                    exc,
                    extra={"event": "position_manager_flat_hook_error"},
                )
    if not old_keys and not new_keys and not snapshot_realized_seen:
        return
    self.save_state()
    self._logger.info(
        "POSITION_SYNC_COMMITTED total=%s added=%s removed=%s "
        "realized_authoritative=%s",
        len(reconciled),
        len(added_symbols),
        len(removed_symbols),
        snapshot_realized_seen,
        extra={
            "event": "POSITION_SYNC_COMMITTED",
            "total_managed": len(reconciled),
            "added": len(added_symbols),
            "removed": len(removed_symbols),
            "realized_pnl_authoritative": snapshot_realized_seen,
        },
    )
''',
)

assert_parses(PATH)
print("patched position manager part A")
