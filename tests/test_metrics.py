from src.diagnostics.metrics import Metrics


def test_metrics_snapshot_updates() -> None:
    m = Metrics()
    m.inc_ticks()
    m.inc_signal()
    m.inc_orders(placed=1)
    m.set_queue_depth(5)
    m.observe_latency(100.0)
    snap = m.snapshot()
    assert snap["signals"] == 1
    assert snap["orders_placed"] == 1
    assert snap["queue_depth"] == 5
    assert snap["avg_latency_ms"] == 100.0
    assert snap["ticks_per_sec"] > 0
