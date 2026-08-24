from pathlib import Path


def replace_once(path: str, old: str, new: str, *, label: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one fixture, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "tests/test_main_health_readiness.py",
    '''    ctx = _ctx(\n        broker_balance_valid=True,\n        evaluation_ready=True,\n        position_reconciliation_completed=True,\n    )\n    main.app.state.bot = SimpleNamespace(_ctx=ctx)\n\n    body = _json(main.health_trading())\n\n    assert body["broker"]["funds_endpoint_verified"] is True\n''',
    '''    ctx = _ctx(\n        broker_balance_valid=True,\n        evaluation_ready=True,\n        live_orders_armed=True,\n        position_reconciliation_completed=True,\n    )\n    main.app.state.bot = SimpleNamespace(_ctx=ctx)\n\n    body = _json(main.health_trading())\n\n    assert body["broker"]["funds_endpoint_verified"] is True\n''',
    label="isolated broker-readiness fixture",
)

# Canonical live arming now consumes the same reconciliation-age contract as
# HTTP health. These fixtures intend to model a successful *fresh* reconcile,
# so give them the timestamp that production reconciliation always records.
replace_once(
    "tests/core/test_minimum_lot_affordability_readiness.py",
    '''        position_reconciliation_completed=True,\n        position_reconciliation_failed=False,\n''',
    '''        position_reconciliation_completed=True,\n        position_reconciliation_completed_at=datetime.now(UTC),\n        position_reconciliation_failed=False,\n''',
    label="minimum-lot fresh reconciliation fixture",
)

replace_once(
    "tests/test_market_open_rearm_loop.py",
    '''        broker_balance_valid=True,\n        position_reconciliation_completed=True,\n        position_reconciliation_failed=False,\n        active_contract_basket={\n''',
    '''        broker_balance_valid=True,\n        position_reconciliation_completed=True,\n        position_reconciliation_completed_at=app.datetime.now(app.timezone.utc),\n        position_reconciliation_failed=False,\n        active_contract_basket={\n''',
    label="rearm fast-path fresh reconciliation fixture",
)
