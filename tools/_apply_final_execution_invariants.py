from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _write(path: str, content: str) -> None:
    (ROOT / path).write_text(content, encoding="utf-8")


def _find_method(source: str, class_name: str, method_name: str) -> ast.AST:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    return child
    raise RuntimeError(f"Unable to find {class_name}.{method_name}")


def replace_method(path: str, class_name: str, method_name: str, replacement: str) -> None:
    source = _read(path)
    node = _find_method(source, class_name, method_name)
    lines = source.splitlines(keepends=True)
    rendered = textwrap.indent(textwrap.dedent(replacement).strip("\n"), "    ") + "\n"
    lines[node.lineno - 1 : node.end_lineno] = [rendered]
    _write(path, "".join(lines))


def replace_in_method(
    path: str,
    class_name: str,
    method_name: str,
    old: str,
    new: str,
) -> None:
    source = _read(path)
    node = _find_method(source, class_name, method_name)
    lines = source.splitlines(keepends=True)
    method_source = "".join(lines[node.lineno - 1 : node.end_lineno])
    count = method_source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}:{class_name}.{method_name}: expected one anchor, found {count}: {old[:120]!r}"
        )
    lines[node.lineno - 1 : node.end_lineno] = [method_source.replace(old, new, 1)]
    _write(path, "".join(lines))


def patch_bracket_flat_verification() -> None:
    replace_method(
        "src/nifty_scalper_bot/execution/bracket_core.py",
        "BracketManager",
        "_verify_position_closed",
        '''
def _verify_position_closed(self, symbol: str) -> bool:
    """Return true only when a valid broker snapshot authoritatively proves flatness."""
    normalized = normalize_symbol(symbol)
    broker = getattr(self.order_manager, "_broker", None)
    if broker is None:
        LOGGER.error(
            "POSITION_FLAT_VERIFY_UNAVAILABLE symbol=%s reason=broker_missing",
            normalized,
            extra={"event": "POSITION_FLAT_VERIFY_UNAVAILABLE", "symbol": normalized},
        )
        return False
    getter = getattr(broker, "get_positions", None)
    if not callable(getter):
        LOGGER.error(
            "POSITION_FLAT_VERIFY_UNAVAILABLE symbol=%s reason=get_positions_missing",
            normalized,
            extra={"event": "POSITION_FLAT_VERIFY_UNAVAILABLE", "symbol": normalized},
        )
        return False

    try:
        response = getter()
    except Exception as exc:  # noqa: BLE001 - unknown exposure must fail closed
        LOGGER.error(
            "POSITION_FLAT_VERIFY_FAILED symbol=%s error=%s",
            normalized,
            exc,
            extra={
                "event": "POSITION_FLAT_VERIFY_FAILED",
                "symbol": normalized,
                "error_type": type(exc).__name__,
            },
        )
        return False

    if response is None:
        LOGGER.error(
            "POSITION_FLAT_VERIFY_INVALID symbol=%s reason=missing_snapshot",
            normalized,
            extra={"event": "POSITION_FLAT_VERIFY_INVALID", "symbol": normalized},
        )
        return False

    if isinstance(response, Mapping):
        rows: object | None = None
        for key in ("net", "positions"):
            if key in response:
                rows = response.get(key)
                break
        if rows is None and any(
            key in response for key in ("symbol", "tradingsymbol", "instrument")
        ):
            rows = [response]
    else:
        rows = response

    if isinstance(rows, (str, bytes, Mapping)) or rows is None:
        return False
    try:
        positions = list(rows)
    except TypeError:
        return False

    for index, position in enumerate(positions):
        if not isinstance(position, Mapping):
            LOGGER.error(
                "POSITION_FLAT_VERIFY_INVALID symbol=%s reason=row_type index=%s",
                normalized,
                index,
                extra={"event": "POSITION_FLAT_VERIFY_INVALID", "symbol": normalized},
            )
            return False
        raw_symbol = (
            position.get("symbol")
            or position.get("tradingsymbol")
            or position.get("instrument")
        )
        if not raw_symbol:
            return False
        position_symbol = normalize_symbol(str(raw_symbol))
        quantity_key = next(
            (
                key
                for key in ("quantity", "net_quantity", "net_qty", "netQuantity")
                if key in position
            ),
            None,
        )
        if quantity_key is None:
            return False
        try:
            quantity = int(float(position.get(quantity_key)))
        except (TypeError, ValueError):
            return False
        if position_symbol == normalized and quantity != 0:
            return False
    return True
''',
    )


def patch_position_snapshot_invariants() -> None:
    path = "src/nifty_scalper_bot/execution/position_manager.py"
    replace_in_method(
        path,
        "PositionManager",
        "synchronize_with_broker",
        """        reconciled: Dict[str, Position] = {}\n        snapshot_realized_pnl = 0.0\n""",
        """        reconciled: Dict[str, Position] = {}\n        seen_symbols: set[str] = set()\n        snapshot_realized_pnl = 0.0\n""",
    )
    replace_in_method(
        path,
        "PositionManager",
        "synchronize_with_broker",
        """            product = str(record.get(\"product\") or \"\").strip().upper()\n            if product != \"MIS\":\n                if symbol in existing_positions:\n                    raise ValueError(\n                        f\"managed broker position {symbol} has unexpected product {product or 'missing'}\"\n                    )\n                continue\n\n            quantity = self._safe_get_net_qty(record)\n""",
        """            product = str(record.get(\"product\") or \"\").strip().upper()\n            if product != \"MIS\":\n                if symbol in existing_positions:\n                    raise ValueError(\n                        f\"managed broker position {symbol} has unexpected product {product or 'missing'}\"\n                    )\n                continue\n            if symbol in seen_symbols:\n                raise ValueError(f\"duplicate broker position row for {symbol}\")\n            seen_symbols.add(symbol)\n\n            quantity = self._safe_get_net_qty(record)\n""",
    )
    replace_method(
        path,
        "PositionManager",
        "_persist_positions_snapshot",
        '''
def _persist_positions_snapshot(self) -> None:
    """Persist position and order snapshots captured from one locked state instant."""
    manager = self._persistent_state
    if manager is None:
        return
    with self._lock:
        position_snapshot = [
            position.to_dict() for position in self._positions.values()
        ]
        order_snapshot = [order.to_dict() for order in self._orders.values()]

    current_symbols = {
        str(entry.get("symbol", "")).strip().upper()
        for entry in position_snapshot
        if str(entry.get("symbol", "")).strip()
    }
    try:
        stored = manager.load_positions()
    except Exception as exc:  # noqa: BLE001
        self._logger.error("Failure in _persist_positions_snapshot: %s", exc)
        stored = []
    stored_symbols = {
        str(item.get("symbol", "")).strip().upper()
        for item in stored
        if isinstance(item, Mapping)
    }
    for entry in position_snapshot:
        try:
            manager.save_position(entry)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _persist_positions_snapshot save: %s", exc
            )
    for symbol in stored_symbols - current_symbols:
        try:
            manager.save_position({"symbol": symbol, "quantity": 0})
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _persist_positions_snapshot remove: %s", exc
            )
    for payload in order_snapshot:
        try:
            manager.save_order(payload)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in _persist_positions_snapshot order save: %s", exc
            )
    try:
        manager.flush()
    except Exception as exc:  # noqa: BLE001
        self._logger.error("Failure in _persist_positions_snapshot flush: %s", exc)
''',
    )


def append_tests() -> None:
    path = ROOT / "tests/execution/test_execution_safety_audit_fixes.py"
    source = path.read_text(encoding="utf-8")
    source += '''

def test_flat_verification_rejects_missing_and_malformed_snapshots() -> None:
    for response in (None, [None], [{"symbol": SYMBOL}], [{"quantity": 0}]):
        broker = SimpleNamespace(get_positions=lambda response=response: response)
        manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
        _stop(manager)
        assert manager._verify_position_closed(SYMBOL) is False


def test_flat_verification_accepts_only_valid_explicit_flat_snapshot() -> None:
    broker = SimpleNamespace(
        get_positions=lambda: [
            {"symbol": SYMBOL, "quantity": 0},
            {"symbol": "NFO:NIFTY2662324000CE", "quantity": 65},
        ]
    )
    manager = BracketManager(order_manager=SimpleNamespace(_broker=broker))
    _stop(manager)
    assert manager._verify_position_closed(SYMBOL) is True

    broker.get_positions = lambda: [{"symbol": SYMBOL, "quantity": 65}]
    assert manager._verify_position_closed(SYMBOL) is False


def test_duplicate_managed_position_rows_reject_snapshot_atomically(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    duplicate = {
        "symbol": SYMBOL,
        "product": "MIS",
        "quantity": 65,
        "average_price": 100.0,
        "last_price": 150.0,
    }
    with pytest.raises(ValueError, match="duplicate broker position"):
        manager.synchronize_with_broker([duplicate, dict(duplicate)])
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)
'''
    path.write_text(source, encoding="utf-8")


def main() -> None:
    patch_bracket_flat_verification()
    patch_position_snapshot_invariants()
    append_tests()


if __name__ == "__main__":
    main()
