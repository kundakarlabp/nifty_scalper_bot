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


def replace_once(path: str, old: str, new: str) -> None:
    source = _read(path)
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    _write(path, source.replace(old, new, 1))


def patch_position_manager() -> None:
    path = "src/nifty_scalper_bot/execution/position_manager.py"
    replace_once(
        path,
        """        with self._lock:\n            existing_positions = dict(self._positions)\n\n    def _get_float(\n""",
        """        with self._lock:\n            existing_positions = copy.deepcopy(self._positions)\n\n    def _get_float(\n""",
    )
    replace_once(
        path,
        """        except Exception as exc:  # noqa: BLE001\n            self._logger.error(\n                \"Failure in _restore_from_persistent_manager positions: %s\",\n                exc,\n            )\n            payloads = []\n        self.restore_positions(payloads)\n""",
        """        except Exception as exc:  # noqa: BLE001\n            self._logger.error(\n                \"Failure in _restore_from_persistent_manager positions: %s\",\n                exc,\n            )\n            return\n        self.restore_positions(payloads)\n""",
    )


def patch_zerodha_positions() -> None:
    path = "src/nifty_scalper_bot/data/rest/zerodha_client.py"
    replace_method(
        path,
        "ZerodhaKiteClient",
        "get_positions",
        '''
def get_positions(self) -> list[dict[str, Any]]:
    """Return one authoritative Zerodha net-position snapshot.

    Missing or malformed ``data.net`` is an error, not an empty account. This
    prevents broker/schema failures from being interpreted as confirmed flatness.
    """
    LOGGER.debug(
        "Entered ZerodhaKiteClient.get_positions",
        extra={"event": "zerodha_get_positions_start"},
    )
    label = "positions.fetch"
    endpoint = "/portfolio/positions"
    should_retry, on_retry = self._build_retry_handlers(endpoint=endpoint)

    def _operation() -> list[dict[str, Any]]:
        with _BROKER_SYNC_LOCK:
            self._acquire_bucket(self._GENERAL_BUCKET)
            response = self._ensure_json(
                self._make_request(
                    "GET", endpoint, operation_label=label, with_retry=False
                )
            )
        payload = response.get("data")
        if not isinstance(payload, Mapping):
            raise BrokerError("Malformed positions response: data object missing")
        net_positions = payload.get("net")
        if not isinstance(net_positions, list):
            raise BrokerError("Malformed positions response: data.net list missing")

        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(net_positions):
            if not isinstance(row, Mapping):
                raise BrokerError(
                    f"Malformed positions response: data.net[{index}] is not an object"
                )
            normalized.append(dict(row))

        if normalized:
            LOGGER.info(
                "zerodha_positions_fetch_success count=%d",
                len(normalized),
                extra={
                    "event": "zerodha_positions_fetch_success",
                    "count": len(normalized),
                },
            )
        else:
            LOGGER.debug("zerodha_positions_fetch_success count=0")
        self._positions_cache = _RestCacheEntry(
            payload=list(normalized),
            updated_at=self._log_time_fn(),
        )
        return normalized

    try:
        return self._execute_with_retry(
            label=label,
            operation=_operation,
            should_retry=should_retry,
            error_message="Failed to fetch Zerodha positions",
            on_retry=on_retry,
        )
    except Exception as exc:
        LOGGER.error(
            "Failure in ZerodhaKiteClient.get_positions: %s",
            exc,
            extra={"event": "zerodha_get_positions_error"},
            exc_info=exc,
        )
        cached = self._load_rest_cache(self._positions_cache, label=label)
        if cached is not None:
            return cast(list[dict[str, Any]], cached)
        raise
''',
    )


def append_tests() -> None:
    path = ROOT / "tests/execution/test_execution_safety_audit_fixes.py"
    source = path.read_text(encoding="utf-8")
    marker = "from nifty_scalper_bot.execution.position_manager import Order, Position, PositionManager\n"
    replacement = marker + "from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient\nfrom nifty_scalper_bot.utils.errors import BrokerError\n"
    if source.count(marker) != 1:
        raise RuntimeError("test import anchor changed")
    source = source.replace(marker, replacement, 1)
    source += '''

def test_invalid_later_row_does_not_partially_mutate_existing_position(tmp_path) -> None:
    manager = _position_manager(tmp_path)
    original = manager.get_position(SYMBOL)
    assert original is not None
    original_price = original.current_price
    with pytest.raises(ValueError):
        manager.synchronize_with_broker(
            [
                {
                    "symbol": SYMBOL,
                    "product": "MIS",
                    "quantity": 65,
                    "average_price": 100.0,
                    "last_price": 150.0,
                },
                {
                    "symbol": "NFO:NIFTY2662324000CE",
                    "product": "MIS",
                    "quantity": "invalid",
                    "average_price": 100.0,
                    "last_price": 101.0,
                },
            ]
        )
    preserved = manager.get_position(SYMBOL)
    assert preserved is not None
    assert preserved.current_price == pytest.approx(original_price)


def _zerodha_positions_client(response):
    client = object.__new__(ZerodhaKiteClient)
    client._GENERAL_BUCKET = "general"
    client._positions_cache = None
    client._log_time_fn = lambda: 0.0
    client._acquire_bucket = lambda *_args, **_kwargs: None
    client._make_request = lambda *_args, **_kwargs: response
    client._ensure_json = lambda payload: payload
    client._build_retry_handlers = lambda **_kwargs: (lambda *_args, **_kw: False, None)
    client._execute_with_retry = lambda **kwargs: kwargs["operation"]()
    client._load_rest_cache = lambda *_args, **_kwargs: None
    return client


def test_zerodha_missing_net_snapshot_raises_instead_of_returning_flat() -> None:
    client = _zerodha_positions_client({"status": "success", "data": {}})
    with pytest.raises(BrokerError):
        client.get_positions()


def test_zerodha_authoritative_empty_net_does_not_fall_back_to_day_rows() -> None:
    client = _zerodha_positions_client(
        {
            "status": "success",
            "data": {
                "net": [],
                "day": [{"symbol": SYMBOL, "quantity": 65}],
            },
        }
    )
    assert client.get_positions() == []
'''
    path.write_text(source, encoding="utf-8")


def main() -> None:
    patch_position_manager()
    patch_zerodha_positions()
    append_tests()


if __name__ == "__main__":
    main()
