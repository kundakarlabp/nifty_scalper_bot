from pathlib import Path

path = Path("tests/test_main_health_readiness.py")
text = path.read_text(encoding="utf-8")
old = '''    ctx = _ctx(\n        broker_balance_valid=True,\n        evaluation_ready=True,\n        position_reconciliation_completed=True,\n    )\n    main.app.state.bot = SimpleNamespace(_ctx=ctx)\n\n    body = _json(main.health_trading())\n\n    assert body["broker"]["funds_endpoint_verified"] is True\n'''
new = '''    ctx = _ctx(\n        broker_balance_valid=True,\n        evaluation_ready=True,\n        live_orders_armed=True,\n        position_reconciliation_completed=True,\n    )\n    main.app.state.bot = SimpleNamespace(_ctx=ctx)\n\n    body = _json(main.health_trading())\n\n    assert body["broker"]["funds_endpoint_verified"] is True\n'''
if text.count(old) != 1:
    raise SystemExit(f"expected one isolated broker-readiness fixture, found {text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
