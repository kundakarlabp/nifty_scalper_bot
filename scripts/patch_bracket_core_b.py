from __future__ import annotations

from scripts._execution_patch_utils import assert_parses, method_text, replace_method, replace_once

CORE = "src/nifty_scalper_bot/execution/bracket_core.py"
CANONICAL = "src/nifty_scalper_bot/execution/canonical_bracket_manager.py"
RUNTIME = "src/nifty_scalper_bot/execution/runtime_bracket_manager.py"

replace_once(
    CORE,
    "    def _get_storage_path(self) -> Path:\n",
    method_text('''
def _decode_restored_bracket(
    self,
    entry_id: str,
    payload: Mapping[str, Any],
) -> BracketState:
    def finite_float(name: str, value: object, minimum: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{entry_id}: invalid {name}") from exc
        if not math.isfinite(result) or result < minimum:
            raise ValueError(f"{entry_id}: invalid {name}")
        return result

    stored_id = str(payload.get("entry_order_id") or entry_id).strip()
    if stored_id != str(entry_id):
        raise ValueError(f"{entry_id}: entry id mismatch")
    symbol = normalize_symbol(str(payload.get("symbol") or ""))
    if not symbol:
        raise ValueError(f"{entry_id}: symbol missing")
    quantity = int(payload.get("quantity") or 0)
    remaining = int(payload.get("remaining_quantity", quantity) or 0)
    if quantity <= 0 or remaining < 0 or remaining > quantity:
        raise ValueError(f"{entry_id}: invalid quantity state")
    trailing_config = payload.get("trailing_config") or {}
    if not isinstance(trailing_config, Mapping):
        raise ValueError(f"{entry_id}: invalid trailing config")
    raw_targets = payload.get("tp_levels") or []
    if isinstance(raw_targets, (str, bytes, Mapping)):
        raise ValueError(f"{entry_id}: invalid target list")
    targets: list[TargetLevel] = []
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, Mapping):
            raise ValueError(f"{entry_id}: invalid target {index}")
        target_qty = int(raw_target.get("quantity") or 0)
        target_price = finite_float("target price", raw_target.get("price"))
        if target_qty <= 0 or target_qty > quantity or target_price <= 0:
            raise ValueError(f"{entry_id}: invalid target {index}")
        targets.append(
            TargetLevel(
                price=target_price,
                quantity=target_qty,
                executed=bool(raw_target.get("executed", False)),
                name=str(raw_target.get("name") or "TP"),
            )
        )
    entry_price = finite_float("entry price", payload.get("entry_price"))
    state = BracketState(
        entry_order_id=stored_id,
        symbol=symbol,
        side=str(payload.get("side") or ""),
        quantity=quantity,
        entry_price=entry_price,
        sl_trigger_price=finite_float("stop loss", payload.get("sl_trigger_price", 0.0)),
        tp_trigger_price=finite_float("take profit", payload.get("tp_trigger_price", 0.0)),
        remaining_quantity=remaining,
        tp_levels=targets,
        is_virtual=bool(payload.get("is_virtual", True)),
        active=bool(payload.get("active", True)),
        trailing_enabled=bool(payload.get("trailing_enabled", False)),
        trailing_config=dict(trailing_config),
        virtual_sl_id=str(payload.get("virtual_sl_id") or f"vsl_{stored_id}"),
        highest_ltp=finite_float("highest ltp", payload.get("highest_ltp", entry_price)),
        lowest_ltp=finite_float("lowest ltp", payload.get("lowest_ltp", entry_price)),
        last_ltp=finite_float("last ltp", payload.get("last_ltp", entry_price)),
        previous_ltp=finite_float(
            "previous ltp",
            payload.get("previous_ltp", payload.get("last_ltp", entry_price)),
        ),
        tag=payload.get("tag"),
        created_at=finite_float("created at", payload.get("created_at", time.time())),
        updated_at=finite_float("updated at", payload.get("updated_at", time.time())),
        exit_executed=bool(payload.get("exit_executed", False)),
        pending_exit_order_id=payload.get("pending_exit_order_id"),
        exit_in_progress=bool(payload.get("exit_in_progress", False)),
        entry_confirmed=bool(payload.get("entry_confirmed", payload.get("active", True))),
        monitoring_only=bool(payload.get("monitoring_only", False)),
        entry_status=str(
            payload.get("entry_status")
            or ("ACTIVE" if payload.get("active", True) else "PENDING_ENTRY")
        ),
        exit_state=str(
            payload.get("exit_state")
            or (
                BracketExitLifecycle.OPEN_ACTIVE.value
                if payload.get("active", True)
                else BracketExitLifecycle.OPEN_PENDING_FILL.value
            )
        ),
        exit_order_id=payload.get("exit_order_id") or payload.get("pending_exit_order_id"),
        entry_fill_price=payload.get("entry_fill_price"),
        exit_reason=payload.get("exit_reason"),
        exit_triggered_at=payload.get("exit_triggered_at"),
        exit_attempt_count=int(payload.get("exit_attempt_count", 0) or 0),
        last_exit_attempt_at=payload.get("last_exit_attempt_at"),
        last_exit_error=payload.get("last_exit_error"),
        exit_pending=bool(payload.get("exit_pending", False)),
        next_exit_attempt_at=payload.get("next_exit_attempt_at"),
        last_exit_summary_at=float(payload.get("last_exit_summary_at", 0.0) or 0.0),
        closed_at=payload.get("closed_at"),
        position_flat_confirmed=bool(payload.get("position_flat_confirmed", False)),
        close_source=payload.get("close_source"),
        exit_price=payload.get("exit_price"),
        escalated_at=payload.get("escalated_at"),
        _market_escalation_fired=bool(payload.get("market_escalation_fired", False)),
        _atr_warning_logged=bool(payload.get("atr_warning_logged", False)),
        ledger_realized_pnl=payload.get("ledger_realized_pnl"),
        _ledger_pending_entry_price=payload.get("ledger_pending_entry_price"),
        _ledger_pending_exit_order_id=payload.get("ledger_pending_exit_order_id"),
        _ledger_pending_exit_quantity=int(payload.get("ledger_pending_exit_quantity", 0) or 0),
        _ledger_pending_exit_price=payload.get("ledger_pending_exit_price"),
        _ledger_pending_exit_target=payload.get("ledger_pending_exit_target"),
        _ledger_release_hook_fired=bool(payload.get("ledger_release_hook_fired", False)),
        _filled_exit_sync_started_at=float(payload.get("filled_exit_sync_started_at", 0.0) or 0.0),
        _filled_exit_sync_order_id=payload.get("filled_exit_sync_order_id"),
        _last_exit_reconcile_at=float(payload.get("last_exit_reconcile_at", 0.0) or 0.0),
    )
    return state
''') + "    def _get_storage_path(self) -> Path:\n",
)

replace_method(
    CORE,
    "BracketManager",
    "_get_storage_path",
    '''
def _get_storage_path(self) -> Path:
    """Return durable state storage; ephemeral fallback is forbidden in LIVE."""
    configured = Path(os.getenv("DATA_DIR", "data")) / "virtual_brackets.json"
    try:
        configured.parent.mkdir(parents=True, exist_ok=True)
        probe = configured.parent / f".bracket_write_test_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        durable = not str(configured.parent.resolve()).startswith("/tmp")
        if self._is_live_execution() and not durable:
            raise OSError("LIVE bracket state cannot use ephemeral /tmp storage")
        self._state_storage_path = str(configured)
        self._state_storage_durable = durable
        return configured
    except (OSError, PermissionError) as exc:
        if self._is_live_execution():
            self._state_storage_path = str(configured)
            self._state_storage_durable = False
            raise OSError(f"durable bracket storage unavailable: {configured}") from exc
        fallback = Path("/tmp") / "virtual_brackets.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        self._state_storage_path = str(fallback)
        self._state_storage_durable = False
        LOGGER.warning(
            "BRACKET_STORAGE_EPHEMERAL_FALLBACK path=%s error=%s",
            fallback,
            exc,
            extra={"event": "BRACKET_STORAGE_EPHEMERAL_FALLBACK", "path": str(fallback)},
        )
        return fallback
''',
)

replace_method(
    CORE,
    "BracketManager",
    "save_state",
    '''
def save_state(self) -> None:
    """Atomically persist one versioned coherent bracket snapshot."""
    temp_path: Path | None = None
    try:
        path = self._get_storage_path()
        with self._lock:
            payload = {
                "schema_version": 2,
                "saved_at": time.time(),
                "brackets": {
                    entry_id: bracket.to_dict()
                    for entry_id, bracket in self._brackets.items()
                },
                "exit_rescue_attempts": dict(
                    getattr(self, "_exit_rescue_attempts", {})
                ),
                "exit_order_open_since": dict(
                    getattr(self, "_exit_order_open_since", {})
                ),
            }
            snapshots = list(self._brackets.values())
        temp_path = path.with_suffix(
            f"{path.suffix}.tmp.{threading.get_ident()}.{time.time_ns()}"
        )
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        self._clear_persistence_degraded()
    except Exception as exc:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        self._mark_persistence_degraded("snapshot_write_failed", exc)
        raise

    for bracket in snapshots:
        self._log_bracket_event(
            "BRACKET_SNAPSHOT",
            bracket,
            meta={
                "active": bracket.active,
                "sl_trigger_price": bracket.sl_trigger_price,
                "tp_trigger_price": bracket.tp_trigger_price,
                "remaining_quantity": bracket.remaining_quantity,
                "exit_executed": bracket.exit_executed,
                "pending_exit_order_id": bracket.pending_exit_order_id,
                "storage_durable": self._state_storage_durable,
            },
        )
''',
)

replace_method(
    CORE,
    "BracketManager",
    "load_state",
    '''
def load_state(self) -> bool:
    """Restore a complete bracket snapshot atomically before watchdog startup."""
    path = self._get_storage_path()
    if not path.exists():
        LOGGER.info("No bracket state file found - starting fresh")
        return False
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("bracket state payload must be an object")
        if "brackets" in decoded:
            version = int(decoded.get("schema_version", 0) or 0)
            if version not in {1, 2}:
                raise ValueError(f"unsupported bracket state schema {version}")
            records = decoded.get("brackets")
            rescue_attempts = decoded.get("exit_rescue_attempts") or {}
            open_since = decoded.get("exit_order_open_since") or {}
        else:
            records = decoded
            rescue_attempts = {}
            open_since = {}
        if not isinstance(records, Mapping):
            raise ValueError("bracket records must be an object")
        if not isinstance(rescue_attempts, Mapping) or not isinstance(open_since, Mapping):
            raise ValueError("bracket recovery maps are invalid")

        temp_brackets: Dict[str, BracketState] = {}
        temp_order_map: Dict[str, str] = {}
        temp_symbol_map: Dict[str, List[str]] = {}
        temp_controllers: Dict[str, Any] = {}
        controller_errors: list[str] = []
        for raw_entry_id, record in records.items():
            entry_id = str(raw_entry_id).strip()
            if not entry_id or not isinstance(record, Mapping):
                raise ValueError(f"invalid bracket record {raw_entry_id!r}")
            bracket = self._decode_restored_bracket(entry_id, record)
            temp_brackets[entry_id] = bracket
            temp_order_map[entry_id] = entry_id
            temp_symbol_map.setdefault(bracket.symbol, []).append(entry_id)
            if bracket.trailing_enabled:
                try:
                    if self._trailing_controller_factory is not None:
                        temp_controllers[entry_id] = self._trailing_controller_factory(bracket)
                    elif (
                        bracket.trailing_config.get("mode") == "ATR"
                        and self._atr_provider
                        and AdaptiveTrailingController
                    ):
                        mult = float(bracket.trailing_config.get("mult", 1.5) or 1.5)
                        spec = TrailingSpec(trail_by=20.0, step=0.25, activation=0.3)
                        temp_controllers[entry_id] = AdaptiveTrailingController(
                            symbol=bracket.symbol,
                            side="LONG" if bracket.side == "BUY" else "SHORT",
                            entry=bracket.entry_price,
                            sl_order_id=bracket.virtual_sl_id,
                            variety="virtual",
                            spec=spec,
                            get_ltp=lambda _symbol, _b=bracket: _b.last_ltp,
                            modify_order=self._virtual_modify_sl,
                            atr_provider=self._atr_provider,
                            journal=MockJournal(),
                            atr_multiplier=mult,
                        )
                except Exception as exc:  # noqa: BLE001
                    controller_errors.append(
                        f"{entry_id}:{type(exc).__name__}:{exc}"
                    )

        parsed_rescue = {str(key): int(value) for key, value in rescue_attempts.items()}
        parsed_open_since = {str(key): float(value) for key, value in open_since.items()}
        with self._lock:
            self._brackets = temp_brackets
            self._order_to_entry = temp_order_map
            self._symbol_map = temp_symbol_map
            self._trailing_controllers = temp_controllers
            if hasattr(self, "_exit_rescue_attempts"):
                self._exit_rescue_attempts = parsed_rescue
            if hasattr(self, "_exit_order_open_since"):
                self._exit_order_open_since = parsed_open_since
        self._clear_persistence_degraded()
        self._recovery_degraded_reason = None
        if controller_errors:
            self._recovery_degraded_reason = "trailing_controller_restore_failed"
            LOGGER.critical(
                "BRACKET_TRAILING_RESTORE_DEGRADED errors=%s",
                controller_errors,
                extra={
                    "event": "BRACKET_TRAILING_RESTORE_DEGRADED",
                    "errors": controller_errors,
                },
            )
        LOGGER.info(
            "BRACKET_STATE_RESTORED count=%s path=%s",
            len(temp_brackets),
            path,
            extra={
                "event": "BRACKET_STATE_RESTORED",
                "count": len(temp_brackets),
                "path": str(path),
            },
        )
        self._resubscribe_restored_brackets()
        return True
    except Exception as exc:
        self._mark_persistence_degraded("snapshot_restore_failed", exc)
        raise
''',
)

replace_method(
    CANONICAL,
    "CanonicalBracketManager",
    "_broker_position_quantity",
    '''
def _broker_position_quantity(self, symbol: str) -> int | None:
    """Return absolute broker quantity, or ``None`` when exposure is unknown."""
    try:
        return self._authoritative_position_quantity(symbol)
    except Exception as exc:  # noqa: BLE001
        _legacy.LOGGER.error(
            "BROKER_POSITION_QUANTITY_UNKNOWN symbol=%s error=%s",
            _legacy.normalize_symbol(symbol),
            exc,
            extra={
                "event": "BROKER_POSITION_QUANTITY_UNKNOWN",
                "symbol": _legacy.normalize_symbol(symbol),
                "error_type": type(exc).__name__,
            },
        )
        return None
''',
)
replace_method(
    CANONICAL,
    "CanonicalBracketManager",
    "_clear_fill_sync_state",
    '''
@staticmethod
def _clear_fill_sync_state(bracket: Any) -> None:
    bracket._filled_exit_sync_started_at = 0.0
    bracket._filled_exit_sync_order_id = None
''',
)
replace_method(
    CANONICAL,
    "CanonicalBracketManager",
    "_fill_sync_grace_expired",
    '''
def _fill_sync_grace_expired(self, bracket: Any, order_id: str) -> bool:
    now = time.time()
    if (
        bracket._filled_exit_sync_order_id != order_id
        or bracket._filled_exit_sync_started_at <= 0
    ):
        bracket._filled_exit_sync_order_id = order_id
        bracket._filled_exit_sync_started_at = now
    return (
        now - bracket._filled_exit_sync_started_at
    ) >= self._filled_position_sync_grace_seconds
''',
)
replace_method(
    RUNTIME,
    "RuntimeBracketManager",
    "_broker_all_positions_flat",
    '''
def _broker_all_positions_flat(self) -> bool | None:
    """Return explicit all-account flatness, or ``None`` when unknowable."""
    try:
        return self._authoritative_position_snapshot().all_flat
    except Exception as exc:  # noqa: BLE001
        _legacy.LOGGER.error(
            "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED error=%s",
            exc,
            extra={
                "event": "FILL_LEDGER_ORPHAN_POSITION_CHECK_FAILED",
                "error_type": type(exc).__name__,
            },
        )
        return None
''',
)

assert_parses(CORE, CANONICAL, RUNTIME)
print("patched bracket core part B")
