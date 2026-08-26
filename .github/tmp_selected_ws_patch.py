from __future__ import annotations

import subprocess
from pathlib import Path


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, check=check)


test_path = Path('tests/streaming/test_stream_supervisor_fallback.py')
test_text = test_path.read_text()
regression = r'''


@pytest.mark.asyncio
async def test_selected_option_missing_active_ws_token_activates_poll_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ce = "NFO:NIFTY2690124300CE"
    pe = "NFO:NIFTY2690124300PE"
    mdm = SimpleNamespace(
        trading_feed_health=MagicMock(
            return_value={
                "futures_fresh": True,
                "options_fresh": True,
                "selected_ce_age_ms": 50,
                "selected_pe_age_ms": 60,
            }
        ),
        data_age_ms=MagicMock(return_value=60),
        _subscribed_tokens=set(),
        _active_tokens=set(),
        _symbol_to_token={ce: 101, pe: 102},
    )
    ctx = SimpleNamespace(
        websocket_manager=SimpleNamespace(is_connected=MagicMock(return_value=True)),
        market_data_manager=mdm,
        selected_ce=ce,
        selected_pe=pe,
        active_symbol_tokens={ce: 101, pe: 102},
    )
    fallback = _Fallback(running=False)

    await app._polling_failover_supervisor_iteration(
        ctx,
        fallback,
        quote_stale_ms=120000,
        degraded_since=0.0,
        recovered_since=None,
        activate_after=0.0,
    )

    fallback.set_websocket_mode.assert_called_once_with(False)
    fallback.start.assert_called_once()
'''
if 'test_selected_option_missing_active_ws_token_activates_poll_recovery' not in test_text:
    test_path.write_text(test_text.rstrip() + regression.rstrip() + '\n')

red = run(
    'python', '-m', 'pytest', '-q',
    'tests/streaming/test_stream_supervisor_fallback.py',
    '-k', 'selected_option_missing_active_ws_token',
    check=False,
)
if red.returncode == 0:
    raise SystemExit('regression unexpectedly passed before source fix')

source_path = Path('src/nifty_scalper_bot/core/polling_failover_runtime.py')
source = source_path.read_text()
anchor = '''def _resolve_market_open_callable(ctx: Any, app_module: Any | None = None) -> Any:\n    ctx_hook = getattr(ctx, "is_market_open_now", None)\n    if ctx_hook is not None:\n        return ctx_hook\n    module = app_module or _APP_MODULE_REF or sys.modules.get(_APP_MODULE_NAME)\n    return getattr(module, "is_market_open_now", None)\n\n\n'''
helper = '''def _selected_ws_delivery_missing(ctx: Any, mdm: Any) -> tuple[str, ...]:\n    """Return selected options lacking an active broker-WebSocket token."""\n\n    if mdm is None:\n        return ()\n    token_map = dict(getattr(ctx, "active_symbol_tokens", {}) or {})\n    mdm_token_map = dict(getattr(mdm, "_symbol_to_token", {}) or {})\n    active_tokens = set(getattr(mdm, "_subscribed_tokens", set()) or set()) | set(\n        getattr(mdm, "_active_tokens", set()) or set()\n    )\n    missing: list[str] = []\n    for raw_symbol in (\n        getattr(ctx, "selected_ce", None),\n        getattr(ctx, "selected_pe", None),\n    ):\n        symbol = str(raw_symbol or "").strip()\n        if not symbol:\n            continue\n        token = token_map.get(symbol, mdm_token_map.get(symbol))\n        try:\n            token_int = int(token) if token is not None else None\n        except (TypeError, ValueError):\n            token_int = None\n        if token_int is not None and token_int not in active_tokens:\n            missing.append(symbol)\n    return tuple(sorted(set(missing)))\n\n\n'''
if helper not in source:
    if anchor not in source:
        raise SystemExit('helper anchor not found')
    source = source.replace(anchor, anchor + helper, 1)
old = '''    feed_health = _safe_feed_health(mdm)\n    data_age_ms = _safe_data_age_ms(mdm)\n    # Canonical live readiness is stricter than transport-arrival health.  If it\n'''
new = '''    feed_health = _safe_feed_health(mdm)\n    data_age_ms = _safe_data_age_ms(mdm)\n    selected_ws_missing = _selected_ws_delivery_missing(ctx, mdm) if ws_ok else ()\n    if selected_ws_missing:\n        # A globally connected WebSocket is not sufficient when the selected\n        # contracts themselves are absent from the active broker token set.\n        # Keep REST recovery running until those exact live-entry contracts are\n        # restored to WebSocket delivery. Fresh REST/direct quotes must not\n        # masquerade as selected-option WebSocket health.\n        feed_health = dict(feed_health)\n        stale_required = set(feed_health.get("stale_required_symbols") or ())\n        stale_required.update(selected_ws_missing)\n        feed_health["stale_required_symbols"] = sorted(stale_required)\n        feed_health["required_symbol_recovery_active"] = True\n        feed_health["selected_ws_delivery_missing_symbols"] = list(selected_ws_missing)\n    # Canonical live readiness is stricter than transport-arrival health.  If it\n'''
if old not in source:
    raise SystemExit('supervisor anchor not found')
source_path.write_text(source.replace(old, new, 1))

run('python', '-m', 'pytest', '-q',
    'tests/streaming/test_stream_supervisor_fallback.py',
    'tests/core/test_polling_failover.py',
    'tests/streaming/test_futures_live_tick_recovery_deadlock.py')
run('python', '-m', 'compileall', '-q', 'src')
run('git', 'diff', '--check')
run('git', 'config', 'user.name', 'github-actions[bot]')
run('git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
run('git', 'add', str(source_path), str(test_path))
run('git', 'rm', '.github/workflows/tmp-selected-ws-fallback-tdd.yml', '.github/tmp_selected_ws_patch.py')
run('git', 'commit', '-m', 'fix(streaming): recover selected options missing from active websocket')
run('git', 'push')
