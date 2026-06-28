from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
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
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:120]!r}")
    _write(path, source.replace(old, new, 1))


def replace_method(path: str, class_name: str, method_name: str, replacement: str) -> None:
    source = _read(path)
    tree = ast.parse(source)
    target: ast.AST | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == method_name:
                    target = child
                    break
    if target is None:
        raise RuntimeError(f"Unable to find {class_name}.{method_name}")
    lines = source.splitlines(keepends=True)
    rendered = textwrap.indent(textwrap.dedent(replacement).strip("\n"), "    ") + "\n"
    lines[target.lineno - 1 : target.end_lineno] = [rendered]
    _write(path, "".join(lines))


def install_validation_dependencies() -> None:
    """Match the repository's standard CI dependency set."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "-q"],
        cwd=ROOT,
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "dashboard/requirements.txt",
            "-q",
        ],
        cwd=ROOT,
    )


def patch_zerodha_positions() -> None:
    replace_method(
        "src/nifty_scalper_bot/data/rest/zerodha_client.py",
        "ZerodhaKiteClient",
        "get_positions",
        '''
def get_positions(self) -> list[dict[str, Any]]:
    """Return one authoritative Zerodha net-position snapshot.

    Position reconciliation never falls back to cache: a failed or malformed
    broker response is unknown exposure, not confirmed flatness.
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
        raise
''',
    )


def patch_cache_test() -> None:
    path = "tests/data/test_zerodha_rest_cache_fallback.py"
    old = '''def test_get_positions_returns_cached_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use cached positions when REST fetch fails."""

    client = _build_client()
    now = 1_000.0
    monkeypatch.setattr(client, "_log_time_fn", lambda: now)
    client._rest_cache_ttl = 30.0
    client._positions_cache = _RestCacheEntry(
        payload=[{"tradingsymbol": "NIFTY"}],
        updated_at=now - 5.0,
    )

    def _raise(*_: Any, **__: Any) -> None:
        raise BrokerError("boom")

    monkeypatch.setattr(client, "_execute_with_retry", _raise)
    result = client.get_positions()

    assert result == [{"tradingsymbol": "NIFTY"}]
'''
    new = '''def test_get_positions_rejects_cache_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached exposure snapshot must not become reconciliation authority."""

    client = _build_client()
    now = 1_000.0
    monkeypatch.setattr(client, "_log_time_fn", lambda: now)
    client._rest_cache_ttl = 30.0
    client._positions_cache = _RestCacheEntry(
        payload=[{"tradingsymbol": "NIFTY"}],
        updated_at=now - 5.0,
    )

    def _raise(*_: Any, **__: Any) -> None:
        raise BrokerError("boom")

    monkeypatch.setattr(client, "_execute_with_retry", _raise)
    with pytest.raises(BrokerError, match="boom"):
        client.get_positions()
'''
    replace_once(path, old, new)


def main() -> None:
    install_validation_dependencies()
    patch_zerodha_positions()
    patch_cache_test()


if __name__ == "__main__":
    main()
