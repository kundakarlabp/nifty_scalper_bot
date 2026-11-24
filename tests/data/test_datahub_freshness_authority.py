from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.data_hub import DataHub


_SYMBOL = "NFO:NIFTY26AUG24150CE"
_TOKEN = 15779586
_NOW_S = 1_800_000_000.0


class _Mdm:
    def __init__(self, tick: dict | None) -> None:
        self.tick = tick

    def attach_tick_bus(self, _bus) -> None:
        return None

    def get_latest_tick(self, symbol: str):
        if symbol == _SYMBOL and self.tick is not None:
            return dict(self.tick)
        return None


def _tick(age_seconds: float) -> dict:
    ts = datetime.fromtimestamp(_NOW_S - age_seconds, tz=timezone.utc)
    return {
        "symbol": _SYMBOL,
        "instrument_token": _TOKEN,
        "token": _TOKEN,
        "ltp": 100.0,
        "last_price": 100.0,
        "bid": 99.9,
        "ask": 100.1,
        "source": "ws",
        "timestamp": ts.isoformat(),
        "timestamp_quality": "exchange",
    }


def _hub(*, mdm_tick: dict | None, cached_tick: dict) -> DataHub:
    hub = DataHub(_Mdm(mdm_tick), clock=lambda: _NOW_S)
    hub._warmup_grace_s = 0.0
    hub._start_mono = hub._monotonic() - 10.0
    hub._quotes[_SYMBOL] = dict(cached_tick)
    return hub


def test_get_quote_prefers_mdm_ssot_over_older_local_cache() -> None:
    """Runner-facing quote reads must not let a frozen facade cache mask live depth."""
    mdm_tick = _tick(0.2)
    cached_tick = _tick(600.0)
    cached_tick.pop("bid")
    cached_tick.pop("ask")
    hub = _hub(mdm_tick=mdm_tick, cached_tick=cached_tick)

    quote = hub.get_quote(_SYMBOL, allow_pull=False)

    assert quote is not None
    assert quote["bid"] == 99.9
    assert quote["ask"] == 100.1
    assert float(quote["last_tick_age_ms"]) < 1_000.0


def test_is_fresh_prefers_current_mdm_tick_over_frozen_datahub_cache() -> None:
    """Live regression: DataHub cache can freeze while canonical MDM keeps ticking."""
    hub = _hub(mdm_tick=_tick(0.2), cached_tick=_tick(600.0))

    fresh, meta = hub.is_fresh(_SYMBOL, threshold_ms=60_000.0)

    assert fresh is True
    assert float(meta["effective_ms"]) < 1_000.0


def test_is_fresh_does_not_let_fresh_datahub_cache_mask_stale_mdm_truth() -> None:
    """MDM remains authoritative in the opposite direction so the fix is fail-closed."""
    hub = _hub(mdm_tick=_tick(120.0), cached_tick=_tick(0.1))

    fresh, meta = hub.is_fresh(_SYMBOL, threshold_ms=60_000.0)

    assert fresh is False
    assert float(meta["effective_ms"]) >= 120_000.0
    assert meta["reason"] == "stale"


def test_is_fresh_preserves_datahub_cache_fallback_without_mdm_tick() -> None:
    """Legacy/replay contexts without an MDM live tick keep the existing behavior."""
    hub = _hub(mdm_tick=None, cached_tick=_tick(0.1))

    fresh, meta = hub.is_fresh(_SYMBOL, threshold_ms=60_000.0)

    assert fresh is True
    assert float(meta["effective_ms"]) < 1_000.0


class _AsyncAccountMDM:
    def __init__(self, *, live=None, raises=False):
        self.live = live
        self.raises = raises
        self._account_snapshot = {"available": 8000.0}

    def attach_tick_bus(self, _bus):
        return None

    def get_available_balance(self, *, force=False):
        if self.raises:
            raise RuntimeError("broker unavailable")
        return self.live

    async def get_account_snapshot(self, *, force=False):
        return {"available": 9000.0}


def test_force_snapshot_does_not_use_async_mdm_cache_as_live_funds() -> None:
    hub = DataHub(_AsyncAccountMDM(live=None))
    assert hub.get_available_balance(force=True) is None
    assert hub.get_account_snapshot(force=True) == {}
    assert hub.get_account_snapshot(force=False) == {"available": 8000.0}


def test_funds_delegate_failure_does_not_resurrect_cached_balance() -> None:
    hub = DataHub(_AsyncAccountMDM(raises=True))
    assert hub.get_available_balance(force=True) is None
    assert hub.get_available_balance(force=False) is None


def test_real_zero_cash_overrides_stale_positive_snapshot() -> None:
    hub = DataHub(_AsyncAccountMDM(live=0.0))
    assert hub.get_available_balance(force=True) == 0.0


def test_quote_identity_never_relabels_old_basket_token_as_current() -> None:
    old_tick = _tick(0.2)
    old_token = old_tick["instrument_token"]
    new_token = old_token + 1
    hub = _hub(mdm_tick=old_tick, cached_tick=old_tick)
    hub._token_by_symbol[_SYMBOL] = new_token

    quote = hub.get_quote(_SYMBOL, allow_pull=False)

    assert quote is not None
    assert quote["instrument_token"] == old_token
    assert quote["quote_identity_token_mismatch"] is True
    assert quote["tradable_quote"] is False
    assert quote["hard_readiness_eligible"] is False


def test_zero_broker_token_is_invalid_not_missing_identity() -> None:
    hub = _hub(mdm_tick=None, cached_tick={})
    hub._token_by_symbol[_SYMBOL] = 12345

    quote = hub._stamp_quote_identity(_SYMBOL, {"instrument_token": 0, "ltp": 100.0})

    assert quote["instrument_token"] == 0
    assert quote["quote_identity_token_mismatch"] is True
    assert quote["tradable_quote"] is False


def test_quote_identity_prefers_current_mdm_mapping_over_old_facade_token() -> None:
    old_tick = _tick(0.2)
    current_token = _TOKEN + 1
    current_tick = {**old_tick, "instrument_token": current_token, "token": current_token}
    hub = _hub(mdm_tick=current_tick, cached_tick=old_tick)
    hub._token_by_symbol[_SYMBOL] = _TOKEN
    hub._mdm._token_by_symbol = {_SYMBOL: current_token}

    quote = hub.get_quote(_SYMBOL, allow_pull=False)
    assert quote is not None
    assert quote["instrument_token"] == current_token
    assert not quote.get("quote_identity_token_mismatch")

    hub._mdm.tick = old_tick
    quote = hub.get_quote(_SYMBOL, allow_pull=False)
    assert quote is not None
    assert quote["instrument_token"] == _TOKEN
    assert quote["quote_identity_token_mismatch"] is True
    assert quote["hard_readiness_eligible"] is False
    hub.close()
