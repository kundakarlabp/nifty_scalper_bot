from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    source = _read(path)
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:80]!r}")
    _write(path, source.replace(old, new, 1))


def _find_node(source: str, class_name: str, method_name: str) -> ast.AST:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child
    raise RuntimeError(f"Unable to find {class_name}.{method_name}")


def replace_method(path: str, class_name: str, method_name: str, replacement: str) -> None:
    source = _read(path)
    node = _find_node(source, class_name, method_name)
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    rendered = textwrap.dedent(replacement).strip("\n")
    rendered = textwrap.indent(rendered, "    ") + "\n"
    lines[start:end] = [rendered]
    _write(path, "".join(lines))


def replace_in_method(
    path: str,
    class_name: str,
    method_name: str,
    old: str,
    new: str,
) -> None:
    source = _read(path)
    node = _find_node(source, class_name, method_name)
    lines = source.splitlines(keepends=True)
    start = node.lineno - 1
    end = node.end_lineno
    method_source = "".join(lines[start:end])
    count = method_source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}:{class_name}.{method_name}: expected one anchor, found {count}: {old[:80]!r}"
        )
    lines[start:end] = [method_source.replace(old, new, 1)]
    _write(path, "".join(lines))


def patch_position_manager() -> None:
    path = "src/nifty_scalper_bot/execution/position_manager.py"

    replace_in_method(
        path,
        "PositionManager",
        "reconcile_now",
        """            try:\n                previous_positions = copy.deepcopy(self._positions)\n""",
        """            try:\n                with self._lock:\n                    previous_positions = copy.deepcopy(self._positions)\n""",
    )

    replace_in_method(
        path,
        "PositionManager",
        "reconcile_now",
        """            if response is None:\n                payloads = []\n            elif isinstance(response, Mapping):\n                payloads.append(cast(Mapping[str, object], response))\n            elif isinstance(response, Iterable) and not isinstance(\n                response, (str, bytes)\n            ):\n                for item in response:\n                    if isinstance(item, Mapping):\n                        payloads.append(cast(Mapping[str, object], item))\n            else:\n""",
        """            if response is None:\n                reason_token = canonical(\"payload_missing\")\n                self._logger.warning(\n                    \"Position reconciliation returned no authoritative snapshot\",\n                    extra={\"event\": \"position_reconcile_failed\", \"reason\": reason_token},\n                )\n                self._handle_reconcile_failure(\n                    reason=reason_token,\n                    error=None,\n                    payload_count=0,\n                    previous_positions=previous_positions,\n                )\n                return False\n            if isinstance(response, Mapping):\n                rows: object | None = None\n                for container_key in (\"net\", \"positions\", \"day\"):\n                    if container_key in response:\n                        rows = response.get(container_key)\n                        break\n                if rows is None and any(\n                    key in response\n                    for key in (\"tradingsymbol\", \"symbol\", \"instrument\")\n                ):\n                    rows = [response]\n                if rows is None or isinstance(rows, (str, bytes, Mapping)):\n                    reason_token = canonical(\"payload_shape\")\n                    self._handle_reconcile_failure(\n                        reason=reason_token,\n                        error=None,\n                        payload_count=0,\n                        previous_positions=previous_positions,\n                    )\n                    return False\n                try:\n                    raw_rows = list(cast(Iterable[object], rows))\n                except TypeError as exc:\n                    reason_token = canonical(\"payload_shape\")\n                    self._handle_reconcile_failure(\n                        reason=reason_token,\n                        error=exc,\n                        payload_count=0,\n                        previous_positions=previous_positions,\n                    )\n                    return False\n            elif isinstance(response, Iterable) and not isinstance(response, (str, bytes)):\n                raw_rows = list(cast(Iterable[object], response))\n            else:\n                raw_rows = []\n\n            if not isinstance(response, (Mapping, Iterable)) or isinstance(response, (str, bytes)):\n""",
    )

    replace_in_method(
        path,
        "PositionManager",
        "reconcile_now",
        """                return False\n            try:\n                self.synchronize_with_broker(payloads)\n""",
        """                return False\n            for item in raw_rows:\n                if not isinstance(item, Mapping):\n                    reason_token = canonical(\"payload_row_type\")\n                    self._handle_reconcile_failure(\n                        reason=reason_token,\n                        error=None,\n                        payload_count=len(payloads),\n                        previous_positions=previous_positions,\n                    )\n                    return False\n                payloads.append(cast(Mapping[str, object], item))\n            try:\n                self.synchronize_with_broker(payloads)\n""",
    )

    replace_method(
        path,
        "PositionManager",
        "_safe_get_net_qty",
        '''
@staticmethod
def _safe_get_net_qty(record: Mapping[str, object]) -> int:
    """Return broker net quantity without converting missing/invalid data to flat."""
    quantity_keys = ("net_qty", "net_quantity", "netQuantity", "net", "quantity")
    found = False
    for key in quantity_keys:
        if key not in record:
            continue
        found = True
        value = record.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid broker quantity field {key}={value!r}") from exc
    if not found:
        raise ValueError("broker position quantity field missing")
    raise ValueError("broker position quantity is null or invalid")
''',
    )

    replace_method(
        path,
        "PositionManager",
        "synchronize_with_broker",
        '''
def synchronize_with_broker(
    self, broker_positions: Sequence[Mapping[str, object]]
) -> None:
    """Atomically apply a fully validated broker position snapshot.

    An explicit, valid empty sequence is authoritative flat state. Missing,
    malformed, partially decoded, or internally inconsistent snapshots raise and
    leave the last-known-good local state untouched.
    """
    self._logger.debug(
        "Entered synchronize_with_broker",
        extra={"event": "position_manager_sync"},
    )
    try:
        payloads = list(broker_positions)
    except (TypeError, ValueError) as exc:
        raise ValueError("broker position snapshot is not iterable") from exc

    with self._lock:
        existing_positions = dict(self._positions)

    def _get_float(
        record: Mapping[str, object],
        keys: Sequence[str],
        *,
        default: float = 0.0,
    ) -> float:
        for key in keys:
            if key not in record or record.get(key) is None:
                continue
            value = record.get(key)
            try:
                return float(cast(Any, value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid broker numeric field {key}={value!r}") from exc
        return float(default)

    reconciled: Dict[str, Position] = {}
    snapshot_realized_pnl = 0.0
    snapshot_realized_seen = False

    for index, record in enumerate(payloads):
        if not isinstance(record, Mapping):
            raise ValueError(f"broker position row {index} is not a mapping")

        raw_symbol = (
            record.get("tradingsymbol")
            or record.get("symbol")
            or record.get("instrument")
        )
        symbol = str(raw_symbol or "").strip().upper()
        if not symbol:
            raise ValueError(f"broker position row {index} has no symbol")
        if not is_strategy_instrument(symbol):
            continue

        product = str(record.get("product") or "").strip().upper()
        if product != "MIS":
            if symbol in existing_positions:
                raise ValueError(
                    f"managed broker position {symbol} has unexpected product {product or 'missing'}"
                )
            continue

        quantity = self._safe_get_net_qty(record)
        realized_pnl = _get_float(record, ("realised", "realized"), default=0.0)
        if "realised" in record or "realized" in record:
            snapshot_realized_seen = True
            snapshot_realized_pnl += realized_pnl

        if quantity == 0:
            continue

        side: Side = "LONG" if quantity > 0 else "SHORT"
        abs_quantity = abs(quantity)
        entry_price = _get_float(
            record,
            ("average_price", "avg_price", "price", "buy_price"),
        )
        current_price = _get_float(
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
        try:
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
                self._logger.info(
                    "Sync: Imported NEW %s position: %s x %s",
                    side,
                    symbol,
                    abs_quantity,
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
        except Exception as exc:
            raise ValueError(f"failed to apply broker position {symbol}") from exc
        reconciled[symbol] = position

    with self._lock:
        old_keys = set(self._positions)
        new_keys = set(reconciled)
        removed_symbols = sorted(old_keys - new_keys)
        added_symbols = sorted(new_keys - old_keys)
        self._positions = reconciled
        if snapshot_realized_seen:
            self._daily_realized_pnl = float(snapshot_realized_pnl)

    if removed_symbols:
        self._logger.info(
            "Sync: Pruned %s positions not found in broker: %s",
            len(removed_symbols),
            removed_symbols,
        )
        hook = getattr(self, "_on_symbols_flat_hook", None)
        if hook is not None:
            try:
                hook(list(removed_symbols))
            except Exception as exc:  # noqa: BLE001 - observer isolation
                self._logger.error(
                    "Failure in on_symbols_flat hook: %s",
                    exc,
                    extra={"event": "position_manager_flat_hook_error"},
                )

    if not old_keys and not new_keys and not snapshot_realized_seen:
        self._logger.debug("Sync: Broker and local state both empty.")
        return

    self.save_state()
    self._logger.info(
        "Sync: Completed successfully.",
        extra={
            "total_managed": len(reconciled),
            "added": len(added_symbols),
            "removed": len(removed_symbols),
            "realized_pnl_authoritative": snapshot_realized_seen,
        },
    )
''',
    )

    replace_method(
        path,
        "PositionManager",
        "restore_positions",
        '''
def restore_positions(self, payloads: Iterable[Mapping[str, Any]]) -> None:
    """Restore a validated persisted snapshot, including an explicit empty state."""
    self._logger.debug(
        "Entered restore_positions",
        extra={"event": "position_manager_restore"},
    )
    try:
        items = list(payloads)
    except TypeError as exc:
        raise ValueError("persisted position snapshot is not iterable") from exc

    rebuilt: Dict[str, Position] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"persisted position row {index} is not a mapping")
        try:
            position = Position.from_dict(cast(Mapping[str, Any], item))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid persisted position row {index}") from exc
        rebuilt[position.symbol.upper()] = position

    with self._lock:
        self._positions = rebuilt
    self._logger.info(
        "Condition met: restore_positions_applied",
        extra={"event": "position_manager_restore_applied", "count": len(rebuilt)},
    )
    self.save_state()
''',
    )

    replace_method(
        path,
        "PositionManager",
        "update_from_order",
        '''
def update_from_order(self, order: Order) -> None:
    """Apply a confirmed local :class:`Order` through the normal fill lifecycle."""
    if not isinstance(order, Order):
        raise TypeError("update_from_order requires position_manager.Order")
    if order.status != "FILLED":
        return
    if order.filled_quantity <= 0:
        raise ValueError("filled order has no filled quantity")
    fill_price = order.fill_price
    if fill_price is None or float(fill_price) <= 0:
        raise ValueError("filled order has no valid fill_price")
    with self._lock:
        self._orders.setdefault(order.order_id, order)
    self.update_order_status(order.order_id, "FILLED", fill_price=float(fill_price))
''',
    )

    replace_method(
        path,
        "PositionManager",
        "update_position_price",
        '''
def update_position_price(self, symbol: str, current_price: float) -> None:
    """Update the mark price of an open position under the state lock."""
    symbol_key = symbol.upper()
    with self._lock:
        position = self._positions.get(symbol_key)
        if position is None:
            return
        position.current_price = float(current_price)
    self.save_state()
''',
    )

    replace_method(
        path,
        "PositionManager",
        "save_state",
        '''
def save_state(self) -> None:
    """Persist one coherent positions/orders snapshot to disk."""
    with self._lock:
        state = {
            "positions": [position.to_dict() for position in self._positions.values()],
            "orders": [order.to_dict() for order in self._orders.values()],
            "daily_realized_pnl": self._daily_realized_pnl,
            "active_contracts": [
                contract.to_dict() for contract in self._active_contracts.values()
            ],
        }
        reconciled_snapshot = copy.deepcopy(self._positions)
    try:
        _atomic_write_json(self._state_path, state)
    except Exception as exc:  # noqa: BLE001 - handled by callers/diagnostics
        self._logger.error("Failed to save position state: %s", exc)
        return
    self._persist_positions_snapshot()
    with self._lock:
        self._last_reconciled_state = reconciled_snapshot
    self._maybe_flush_persistent_state()
''',
    )

    replace_in_method(
        path,
        "PositionManager",
        "_handle_reconcile_failure",
        """                self._positions = copy.deepcopy(dict(fallback))\n                restore_applied = True\n""",
        """                restored_positions = copy.deepcopy(dict(fallback))\n                with self._lock:\n                    self._positions = restored_positions\n                restore_applied = True\n""",
    )


def patch_bracket_core() -> None:
    path = "src/nifty_scalper_bot/execution/bracket_core.py"

    replace_method(
        path,
        "TargetLevel",
        "__init__",
        "",  # never reached; dataclass has no explicit __init__
    ) if False else None

    replace_once(
        path,
        """class TargetLevel:\n    \"\"\"Represents a partial profit target level.\"\"\"\n    price: float\n    quantity: int\n    executed: bool = False\n    name: str = \"TP\"\n\n\n@dataclass\nclass BracketState:\n""",
        """class TargetLevel:\n    \"\"\"Represents a partial profit target level.\"\"\"\n    price: float\n    quantity: int\n    executed: bool = False\n    name: str = \"TP\"\n\n    def to_dict(self) -> dict[str, Any]:\n        \"\"\"Return a JSON-safe target snapshot.\"\"\"\n        return {\n            \"price\": float(self.price),\n            \"quantity\": int(self.quantity),\n            \"executed\": bool(self.executed),\n            \"name\": str(self.name),\n        }\n\n\n@dataclass\nclass BracketState:\n""",
    )

    replace_once(
        path,
        """    _market_escalation_fired: bool = False\n    _atr_warning_logged: bool = False\n\n    @property\n""",
        """    _market_escalation_fired: bool = False\n    _atr_warning_logged: bool = False\n    ledger_realized_pnl: dict[str, Any] | None = None\n    _ledger_pending_entry_price: float | None = None\n    _ledger_pending_exit_order_id: str | None = None\n    _ledger_pending_exit_quantity: int = 0\n    _ledger_pending_exit_price: float | None = None\n    _ledger_pending_exit_target: str | None = None\n    _ledger_release_hook_fired: bool = False\n\n    @property\n""",
    )

    replace_in_method(
        path,
        "BracketState",
        "__post_init__",
        """        # Auto-initialize state fields if not set\n""",
        """        self.side = _normalize_bracket_side(self.side)\n        # Auto-initialize state fields if not set\n""",
    )

    replace_in_method(
        path,
        "BracketState",
        "to_dict",
        """            \"lowest_ltp\": self.lowest_ltp,\n            \"tag\": self.tag,\n""",
        """            \"lowest_ltp\": self.lowest_ltp,\n            \"last_ltp\": self.last_ltp,\n            \"tag\": self.tag,\n""",
    )
    replace_in_method(
        path,
        "BracketState",
        "to_dict",
        """            \"atr_warning_logged\": self._atr_warning_logged,\n""",
        """            \"atr_warning_logged\": self._atr_warning_logged,\n            \"ledger_realized_pnl\": self.ledger_realized_pnl,\n            \"ledger_pending_entry_price\": self._ledger_pending_entry_price,\n            \"ledger_pending_exit_order_id\": self._ledger_pending_exit_order_id,\n            \"ledger_pending_exit_quantity\": self._ledger_pending_exit_quantity,\n            \"ledger_pending_exit_price\": self._ledger_pending_exit_price,\n            \"ledger_pending_exit_target\": self._ledger_pending_exit_target,\n            \"ledger_release_hook_fired\": self._ledger_release_hook_fired,\n""",
    )

    replace_in_method(
        path,
        "BracketManager",
        "register_virtual_bracket",
        """        symbol = normalize_symbol(symbol)\n""",
        """        symbol = normalize_symbol(symbol)\n        side = _normalize_bracket_side(side)\n""",
    )

    replace_in_method(
        path,
        "BracketManager",
        "register_virtual_bracket",
        """                except Exception as exc:\n                    LOGGER.exception(\n                        '[CRITICAL FAILURE]',\n                        extra={'event': 'bracket_manager_metrics_increment_error'},\n                        exc_info=True,\n                    )\n                    raise\n""",
        """                except Exception as exc:\n                    LOGGER.error(\n                        \"BRACKET_METRICS_INCREMENT_FAILED order_id=%s symbol=%s error=%s\",\n                        order_id,\n                        symbol,\n                        exc,\n                        extra={\n                            \"event\": \"bracket_manager_metrics_increment_error\",\n                            \"order_id\": order_id,\n                            \"symbol\": symbol,\n                            \"error_type\": type(exc).__name__,\n                        },\n                    )\n""",
    )

    replace_in_method(
        path,
        "BracketManager",
        "confirm_entry_fill",
        """            except Exception as e:\n                LOGGER.exception(\"Unhandled exception\", exc_info=True)\n                raise\n""",
        """            except Exception as exc:\n                LOGGER.critical(\n                    \"BRACKET_ACTIVATION_PERSIST_FAILED order_id=%s symbol=%s error=%s\",\n                    order_id,\n                    bracket.symbol,\n                    exc,\n                    extra={\n                        \"event\": \"BRACKET_ACTIVATION_PERSIST_FAILED\",\n                        \"order_id\": order_id,\n                        \"symbol\": bracket.symbol,\n                        \"error_type\": type(exc).__name__,\n                    },\n                )\n""",
    )

    replace_in_method(
        path,
        "BracketManager",
        "on_tick",
        """        symbol = normalize_symbol(symbol)\n        \"\"\"\n        Ultra-low-latency tick processing.\n        \n        ✅ WORLD-CLASS OPTIMIZATIONS:\n        - Minimal lock scope (snapshot only)\n        - No lock during exit evaluation\n        - Batch exit firing\n        - Atomic LTP updates\n        \"\"\"\n""",
        """        \"\"\"Process one tick through trailing and hard SL/TP evaluation.\"\"\"\n        symbol = normalize_symbol(symbol)\n""",
    )

    replace_method(
        path,
        "BracketManager",
        "_get_current_atr",
        '''
def _get_current_atr(self, symbol: str) -> float:
    """Return current ATR, falling back to the validated local cache."""
    if self._atr_provider:
        try:
            atr_value = None
            if hasattr(self._atr_provider, "get_current_atr"):
                atr_value = self._atr_provider.get_current_atr(symbol)
            elif hasattr(self._atr_provider, "get_atr"):
                snapshot = self._atr_provider.get_atr(symbol)
                atr_value = getattr(snapshot, "value", snapshot)
            if atr_value is not None and float(atr_value) > 0:
                return float(atr_value)
        except Exception as exc:  # noqa: BLE001 - hot path must retain SL/TP checks
            self._log_throttled(
                "warning",
                f"atr_provider_failure_{symbol}",
                30.0,
                "ATR_PROVIDER_FAILED_USING_CACHE symbol=%s error_type=%s error=%s",
                symbol,
                type(exc).__name__,
                exc,
            )

    atr_raw = self._current_atr.get(symbol, 0.0)
    if isinstance(atr_raw, Mapping):
        atr_raw = atr_raw.get("value", 0.0)
    try:
        atr_value = float(atr_raw or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return atr_value if math.isfinite(atr_value) and atr_value > 0 else 0.0
''',
    )

    replace_in_method(
        path,
        "BracketManager",
        "_verify_position_closed",
        """            if broker is None:\n                return True\n            getter = getattr(broker, 'get_positions', None)\n            if not callable(getter):\n                return True\n""",
        """            if broker is None:\n                LOGGER.error(\n                    \"POSITION_FLAT_VERIFY_UNAVAILABLE symbol=%s reason=broker_missing\",\n                    symbol,\n                    extra={\"event\": \"POSITION_FLAT_VERIFY_UNAVAILABLE\", \"symbol\": symbol},\n                )\n                return False\n            getter = getattr(broker, 'get_positions', None)\n            if not callable(getter):\n                LOGGER.error(\n                    \"POSITION_FLAT_VERIFY_UNAVAILABLE symbol=%s reason=get_positions_missing\",\n                    symbol,\n                    extra={\"event\": \"POSITION_FLAT_VERIFY_UNAVAILABLE\", \"symbol\": symbol},\n                )\n                return False\n""",
    )

    replace_in_method(
        path,
        "BracketManager",
        "_get_storage_path",
        """            path = Path(os.getenv(\"DATA_DIR\", \"data\")) / \"virtual_brackets.json\"\n            path.parent.mkdir(parents=True, exist_ok=True)\n            LOGGER.warning(f\"⚠️ Using /tmp fallback for brackets: {path}\")\n""",
        """            path = Path(\"/tmp\") / \"virtual_brackets.json\"\n            path.parent.mkdir(parents=True, exist_ok=True)\n            LOGGER.warning(\"Using /tmp fallback for brackets: %s\", path)\n""",
    )

    replace_method(
        path,
        "BracketManager",
        "save_state",
        '''
def save_state(self) -> None:
    """Atomically persist one coherent bracket snapshot and journal it."""
    with self._lock:
        payload = {
            entry_id: bracket.to_dict()
            for entry_id, bracket in self._brackets.items()
        }
        snapshots = list(self._brackets.values())
    path = self._get_storage_path()
    temp_path = path.with_suffix(
        f"{path.suffix}.tmp.{threading.get_ident()}.{time.time_ns()}"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except Exception:
        with suppress(OSError):
            temp_path.unlink()
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
            },
        )
''',
    )

    replace_in_method(
        path,
        "BracketManager",
        "load_state",
        """            restored_count = 0\n            with self._lock:\n                for eid, d in data.items():\n""",
        """            if not isinstance(data, Mapping):\n                raise ValueError(\"bracket state payload must be an object\")\n            restored_count = 0\n            with self._lock:\n                self._brackets.clear()\n                self._order_to_entry.clear()\n                self._symbol_map.clear()\n                self._trailing_controllers.clear()\n                for eid, d in data.items():\n""",
    )
    replace_in_method(
        path,
        "BracketManager",
        "load_state",
        """                        b._atr_warning_logged = bool(d.get(\"atr_warning_logged\", False))\n                        b.virtual_sl_id = f\"vsl_{eid}\"\n""",
        """                        b._atr_warning_logged = bool(d.get(\"atr_warning_logged\", False))\n                        b.ledger_realized_pnl = d.get(\"ledger_realized_pnl\")\n                        b._ledger_pending_entry_price = d.get(\"ledger_pending_entry_price\")\n                        b._ledger_pending_exit_order_id = d.get(\"ledger_pending_exit_order_id\")\n                        b._ledger_pending_exit_quantity = int(d.get(\"ledger_pending_exit_quantity\", 0) or 0)\n                        b._ledger_pending_exit_price = d.get(\"ledger_pending_exit_price\")\n                        b._ledger_pending_exit_target = d.get(\"ledger_pending_exit_target\")\n                        b._ledger_release_hook_fired = bool(d.get(\"ledger_release_hook_fired\", False))\n                        b.virtual_sl_id = f\"vsl_{eid}\"\n""",
    )
    replace_in_method(
        path,
        "BracketManager",
        "load_state",
        """                                        get_ltp=lambda s: b.last_ltp,\n""",
        """                                        get_ltp=lambda s, _b=b: _b.last_ltp,\n""",
    )


def patch_margin_engine() -> None:
    path = "src/nifty_scalper_bot/execution/margin_engine.py"
    replace_once(path, "from dataclasses import dataclass\n", "from dataclasses import dataclass, replace\n")
    replace_once(
        path,
        "@dataclass(slots=True)\nclass MarginInputs:",
        "@dataclass(frozen=True, slots=True)\nclass MarginInputs:",
    )
    replace_in_method(
        path,
        "MarginEngine",
        "plan",
        """        original_balance = inputs.balance\n        inputs.balance = effective_balance\n        try:\n            order_type = self._resolve_order_type(inputs.ist_now, inputs.product)\n            max_units = self._max_qty_from_risk(inputs)\n        finally:\n            inputs.balance = original_balance\n""",
        """        effective_inputs = replace(inputs, balance=effective_balance)\n        order_type = self._resolve_order_type(\n            effective_inputs.ist_now, effective_inputs.product\n        )\n        max_units = self._max_qty_from_risk(effective_inputs)\n""",
    )


def patch_execution_policy() -> None:
    path = "src/nifty_scalper_bot/execution/execution_policy.py"
    replace_once(
        path,
        "    max_spread_pct: float = 0.015\n",
        "    max_spread_pct: float | None = 0.015\n",
    )
    replace_in_method(
        path,
        "ExecutionPolicy",
        "__post_init__",
        """        self.max_spread_pct = max(0.0, float(self.max_spread_pct))\n""",
        """        if self.max_spread_pct is not None:\n            self.max_spread_pct = max(0.0, float(self.max_spread_pct))\n""",
    )
    replace_in_method(
        path,
        "ExecutionPolicy",
        "build_plan",
        """        if effective_max_spread_pct and spread_pct > effective_max_spread_pct:\n""",
        """        if (\n            effective_max_spread_pct is not None\n            and spread_pct > effective_max_spread_pct\n        ):\n""",
    )
    replace_method(
        path,
        "ExecutionPolicy",
        "_effective_max_spread_pct",
        '''
def _effective_max_spread_pct(
    self, symbol: str, last_price: float, mid: float
) -> float | None:
    if self.max_spread_pct is None:
        return None
    base = float(self.max_spread_pct)
    upper = symbol.upper()
    is_option = upper.endswith("CE") or upper.endswith("PE")
    if not is_option:
        return base
    try:
        atm_limit = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_ATM_PCT"), 0.03)
        low_premium_limit = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_LOW_PREMIUM_PCT"), 0.06)
        hard_cap = parse_float_env(os.getenv("OPTION_EXEC_MAX_SPREAD_HARD_CAP_PCT"), 0.10)
        low_premium_cutoff = parse_float_env(os.getenv("OPTION_LOW_PREMIUM_CUTOFF"), 50.0)
    except ValueError:
        atm_limit = 0.03
        low_premium_limit = 0.06
        hard_cap = 0.10
        low_premium_cutoff = 50.0
    reference = float(last_price or mid or 0.0)
    if reference > 0 and reference <= low_premium_cutoff:
        return min(max(base, low_premium_limit), hard_cap)
    return min(max(base, atm_limit), hard_cap)
''',
    )


def write_tests() -> None:
    path = ROOT / "tests/execution/test_execution_safety_audit_fixes.py"
    path.write_text(
        '''from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution import bracket_core
from nifty_scalper_bot.execution.bracket_core import BracketManager, BracketState
from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.execution.margin_engine import MarginInputs
from nifty_scalper_bot.execution.position_manager import Order, Position, PositionManager
from nifty_scalper_bot.utils.errors import OrderPlacementError


SYMBOL = "NFO:NIFTY2662324050PE"


def _position() -> Position:
    return Position(
        symbol=SYMBOL,
        side="LONG",
        quantity=65,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        current_price=101.0,
    )


def _position_manager(tmp_path) -> PositionManager:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager._schedule_retry_after_failure = lambda *_args, **_kwargs: None
    manager._positions[SYMBOL] = _position()
    return manager


def test_none_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: None))

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_malformed_broker_snapshot_fails_closed_and_preserves_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.set_broker_client(
        SimpleNamespace(
            get_positions=lambda: [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": "not-a-number",
                    "average_price": 100.0,
                    "last_price": 101.0,
                }
            ]
        )
    )

    assert manager.reconcile_now() is False
    assert manager.get_position(SYMBOL) is not None


def test_explicit_empty_snapshot_is_authoritative_flat(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    flattened: list[list[str]] = []
    manager.set_on_symbols_flat(lambda symbols: flattened.append(list(symbols)))
    manager.set_broker_client(SimpleNamespace(get_positions=lambda: []))

    assert manager.reconcile_now() is True
    assert manager.get_position(SYMBOL) is None
    assert flattened == [[SYMBOL]]


def test_broker_realised_field_updates_daily_realised_without_using_total_pnl(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    manager.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "product": "MIS",
                "quantity": 0,
                "realised": -125.5,
                "pnl": 9999.0,
                "m2m": 9999.0,
            }
        ]
    )
    assert manager.get_realized_pnl() == pytest.approx(-125.5)


def test_update_from_order_uses_fill_price_and_existing_fill_lifecycle(tmp_path) -> None:
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    order = Order(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        order_type="MARKET",
        quantity=65,
        price=100.0,
        status="FILLED",
        filled_quantity=65,
        fill_price=101.0,
    )
    manager.update_from_order(order)
    position = manager.get_position(SYMBOL)
    assert position is not None
    assert position.entry_price == pytest.approx(101.0)
    assert position.quantity == 65


def _stop(manager: BracketManager) -> None:
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)


def test_direct_long_bracket_registration_normalizes_and_triggers_sl(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="LONG",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.side == "BUY"
    action = manager._evaluate_exit_fast(bracket, 89.0)
    assert action is not None
    assert action["type"] == "SL"


def test_bracket_state_is_written_and_restored_with_ledger_recovery_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket._ledger_pending_exit_order_id = "exit-1"
    bracket._ledger_pending_exit_quantity = 65
    bracket._ledger_pending_exit_price = 89.5
    manager.save_state()

    restored = BracketManager(order_manager=SimpleNamespace())
    _stop(restored)
    restored.load_state()
    restored_bracket = restored.get_bracket("entry-1")
    assert restored_bracket is not None
    assert restored_bracket._ledger_pending_exit_order_id == "exit-1"
    assert restored_bracket._ledger_pending_exit_quantity == 65
    assert restored_bracket._ledger_pending_exit_price == pytest.approx(89.5)


def test_confirmed_fill_remains_active_when_snapshot_persistence_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    monkeypatch.setattr(manager, "save_state", lambda: (_ for _ in ()).throw(OSError("disk")))
    manager.confirm_entry_fill("entry-1", 101.0)
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    assert bracket.active is True
    assert bracket.entry_confirmed is True


def test_metrics_failure_does_not_undo_registered_protection(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    manager = BracketManager(order_manager=SimpleNamespace())
    _stop(manager)

    class _Counter:
        def inc(self) -> None:
            raise RuntimeError("metrics down")

    monkeypatch.setattr(bracket_core, "METRICS_AVAILABLE", True)
    monkeypatch.setattr(bracket_core, "METRICS", SimpleNamespace(brackets_created=_Counter()))
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=True,
    )
    assert manager.get_bracket("entry-1") is not None


def test_margin_inputs_are_immutable() -> None:
    inputs = MarginInputs(
        symbol=SYMBOL,
        side="BUY",
        price=100.0,
        stop_loss=90.0,
        atr=5.0,
        requested_qty=65,
        product="MIS",
        lot_size=65,
        balance=100000.0,
        per_trade_risk_pct=1.0,
        per_trade_cap_pct=10.0,
        margin_factor=1.0,
        margin_buffer=0.95,
        contract_multiplier=1.0,
        ist_now=datetime.now(timezone.utc),
        min_lots_per_trade=1,
        max_lots_per_trade=2,
        atr_multiple=1.5,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.balance = 1.0


class _Hub:
    def __init__(self, quote):
        self.quote = quote

    def get_quote(self, symbol: str, allow_pull: bool = True):
        return dict(self.quote)


def test_zero_spread_limit_is_strict_and_none_explicitly_disables_guard() -> None:
    quote = {"best_bid": 100.0, "best_ask": 101.0}
    with pytest.raises(OrderPlacementError):
        ExecutionPolicy(_Hub(quote), max_spread_pct=0.0).build_plan("NSE:NIFTY", "BUY")
    plan = ExecutionPolicy(_Hub(quote), max_spread_pct=None).build_plan("NSE:NIFTY", "BUY")
    assert plan.spread_pct > 0
''',
        encoding="utf-8",
    )


def main() -> None:
    patch_position_manager()
    patch_bracket_core()
    patch_margin_engine()
    patch_execution_policy()
    write_tests()


if __name__ == "__main__":
    main()
