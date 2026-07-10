from nifty_scalper_bot.data.assess_data import assess_datahub_fresh


class _CanonicalHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def is_fresh(self, symbol: str, *, threshold_ms: float):
        self.calls.append((symbol, threshold_ms))
        return False, {
            "symbol": symbol,
            "reason": "stale",
            "effective_ms": threshold_ms + 1.0,
            "threshold_ms": threshold_ms,
            "source": "historical",
        }

    def get_quote(self, *_args, **_kwargs):
        raise AssertionError("canonical is_fresh should own freshness")


def test_assess_datahub_fresh_delegates_single_symbol_to_datahub_is_fresh() -> None:
    hub = _CanonicalHub()

    ok, detail, meta = assess_datahub_fresh(hub, "NSE:NIFTY", 60_000)

    assert ok is False
    assert detail == "stale"
    assert hub.calls == [("NSE:NIFTY", 60_000.0)]
    assert meta["source"] == "historical"
    assert meta["age_ms"] == 60_001.0
