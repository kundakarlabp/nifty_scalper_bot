from __future__ import annotations

from typing import List

from nifty_scalper_bot.infra.metrics import MetricsCollector


def test_metrics_collector_triggers_alerts() -> None:
    collector = MetricsCollector()
    notifications: List[str] = []

    def _notify(state) -> None:
        notifications.append(f"{state.name}:{state.active}")

    collector.register_notifier(_notify)
    base_time = 1_000.0
    for offset in range(20):
        collector.observe_order_latency(
            order_type="MARKET",
            latency_seconds=0.7,
            now=base_time + offset * 5.0,
        )
    for offset in range(24):
        collector.observe_tick(
            symbol="NIFTY",
            latency_seconds=None,
            staleness_seconds=3.0,
            now=base_time + offset * 5.0,
        )
    for offset in range(12):
        collector.increment_error(
            error_type="order_error",
            now=base_time + offset * 20.0,
        )
    collector.record_queue_depth(symbol="NIFTY", depth=25.0, now=base_time + 5.0)
    collector.record_order_rejection(reason="RMS", now=base_time + 15.0)
    collector.set_pnl_breakdown(book="primary", realized=10.0, unrealized=-2.0)
    alerts = collector.snapshot_alerts()
    assert alerts["OrderLatencyHigh"]["active"] is True
    assert alerts["TickStalenessCritical"]["active"] is True
    assert alerts["ErrorBurst"]["active"] is True
    assert any(item.startswith("OrderLatencyHigh:True") for item in notifications)
    assert any(item.startswith("TickStalenessCritical:True") for item in notifications)
    assert any(item.startswith("ErrorBurst:True") for item in notifications)

    summary = collector.metric_summary()
    assert summary["tick_latency_p95"] is None
    assert summary["order_latency_p95"] and summary["order_latency_p95"] > 0.5
    assert summary["max_tick_staleness"] >= 3.0
    burst_counts = summary["error_events_5m"]
    assert burst_counts["order_error"] >= 11
    assert summary["queue_depth_max"] >= 25.0
    assert summary["queue_depth_avg"] >= 25.0
    assert summary["order_rejections_5m"]["RMS"] >= 1
    assert "primary" in summary["pnl_breakdown"]
    assert summary["error_events_per_minute"]


def test_record_heartbeat_flush_metrics() -> None:
    collector = MetricsCollector()
    base_time = 1_000.0
    collector.record_heartbeat_flush(
        success=True,
        latency_seconds=0.2,
        now=base_time,
    )
    collector.record_heartbeat_flush(
        success=False,
        latency_seconds=0.5,
        now=base_time + 60.0,
    )
    summary = collector.metric_summary()
    flush_counts = summary["heartbeat_flushes_5m"]
    assert flush_counts["success"] >= 1
    assert flush_counts["failure"] >= 1
    assert summary["heartbeat_last_latency"] == 0.5
    assert summary["heartbeat_flush_age"] is not None


def test_strategy_allocation_change_counts_transitions() -> None:
    collector = MetricsCollector()

    class _StubCounter:
        def __init__(self) -> None:
            self.label_calls: list[tuple[str, str]] = []
            self.inc_calls = 0

        def labels(self, strategy: str, direction: str) -> "_StubCounter":
            self.label_calls.append((strategy, direction))
            return self

        def inc(self) -> None:
            self.inc_calls += 1

    stub = _StubCounter()
    collector._strategy_allocation_change = stub  # type: ignore[assignment]
    collector.increment_strategy_allocation_change(
        strategy="alpha", direction="increase"
    )
    collector.increment_strategy_allocation_change(
        strategy="alpha", direction="increase"
    )
    collector.increment_strategy_allocation_change(
        strategy="alpha", direction="decrease"
    )

    assert stub.inc_calls == 2
    assert stub.label_calls == [
        ("alpha", "increase"),
        ("alpha", "decrease"),
    ]
