"""OrderFlow shadow handoff must remain bounded and causal."""

from nifty_scalper_bot.core.orderflow_shadow_bus import (
    publish_orderflow_research,
    read_orderflow_research,
)


def test_shadow_bus_window_and_freshness() -> None:
    symbol = "NFO:TEST999CE"
    publish_orderflow_research(
        symbol, ofi_3s_norm=-3, quote_age_seconds=0.2,
        ofi_updates_3s=6, ofi_span_seconds=2.9, sample_time=1000.0,
    )
    publish_orderflow_research(
        symbol, ofi_3s_norm=-3, quote_age_seconds=5.0,
        ofi_updates_3s=6, ofi_span_seconds=2.9, sample_time=1001.0,
    )
    seen = read_orderflow_research(symbol, since=999, until=1000.5)
    assert len(seen) == 1
    assert seen[0]["fresh"] is True
    assert (
        read_orderflow_research(symbol, since=1000.5, until=1001.5)[0]["fresh"]
        is False
    )


def test_shadow_bus_rejects_nonfinite_ofi() -> None:
    symbol = "NFO:TEST999PE"
    publish_orderflow_research(
        symbol, ofi_3s_norm=float("nan"), quote_age_seconds=0.1,
        ofi_updates_3s=5, ofi_span_seconds=2.8, sample_time=1000,
    )
    assert read_orderflow_research(symbol, since=0, until=1001) == []
