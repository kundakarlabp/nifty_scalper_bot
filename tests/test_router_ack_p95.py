from src.execution.order_executor import OrderExecutor


def test_ack_p95() -> None:
    exe = OrderExecutor(kite=None)
    for ms in [100, 200, 300, 400, 500]:
        exe._record_ack_latency(ms)
    rh = exe.router_health()
    assert rh["ack_p95_ms"] == 400
