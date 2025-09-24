"""Runtime patch supplying synthetic warm-up bars when broker OHLC is empty."""

from __future__ import annotations

import datetime as dt
import logging
import os

log = logging.getLogger(__name__)


def _ist_now() -> dt.datetime:
    """Return current time in IST if ``pytz`` is available."""

    try:
        import pytz  # type: ignore[import-untyped]

        ist = pytz.timezone("Asia/Kolkata")
        return dt.datetime.now(ist)
    except Exception:
        return dt.datetime.utcnow()


def _last_session_bounds_ist(
    now: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Return (start, end) of the previous trading session in IST."""

    now = now or _ist_now()
    day_shift = 1
    wd = now.weekday()
    if wd == 0:
        day_shift = 3  # Monday -> Friday
    elif wd == 6:
        day_shift = 2  # Sunday -> Friday
    elif wd == 5:
        day_shift = 1  # Saturday -> Friday
    prev = now - dt.timedelta(days=day_shift)
    start = prev.replace(hour=9, minute=15, second=0, microsecond=0)
    end = prev.replace(hour=15, minute=30, second=0, microsecond=0)
    return (start.replace(tzinfo=None), end.replace(tzinfo=None))


def _make_synth_df(ltp: float, n: int = 30):
    """Return ``n`` synthetic 1‑minute bars all set to ``ltp``."""

    try:
        import pandas as pd
    except Exception:  # pragma: no cover - defensive
        return None
    now = _ist_now().replace(tzinfo=None)
    idx = [now - dt.timedelta(minutes=i) for i in range(n, 0, -1)]
    data = {
        "date": idx,
        "open": [ltp] * n,
        "high": [ltp] * n,
        "low": [ltp] * n,
        "close": [ltp] * n,
        "volume": [0] * n,
    }
    df = pd.DataFrame(data)
    return df


def _get_last_session_close(kite, token: int) -> float | None:
    """Return previous session's closing price for ``token`` if available."""

    try:
        prev_start, prev_end = _last_session_bounds_ist()
        rows = kite.historical_data(token, prev_start, prev_end, "minute")
        if rows:
            last = rows[-1]
            close = float(last.get("close", 0.0))
            return close if close > 0 else None
    except Exception:
        pass
    return None


def _patch_fetch_ohlc_df() -> None:
    try:
        from src.data import source as ds  # type: ignore
    except Exception:
        log.exception("synthetic_warmup: cannot import src.data.source")
        return
    cls = getattr(ds, "LiveKiteSource", None)
    if not cls:
        return
    orig = getattr(cls, "_fetch_ohlc_df", None)
    if not callable(orig):
        return

    def wrap(self, *a, **k):  # type: ignore[override]
        try:
            df = orig(self, *a, **k)
            if df is not None and getattr(df, "empty", False) is False:
                return df
        except Exception:
            df = None

        token = k.get("token")
        try:
            if token is None and a:
                token = a[0]
        except Exception:
            token = None

        ltp = None
        try:
            if token is not None and hasattr(self, "get_last_price"):
                ltp = self.get_last_price(token)
        except Exception:
            ltp = None
        if not ltp or ltp <= 0:
            try:
                ltp = _get_last_session_close(self.kite, int(token))
            except Exception:
                ltp = None

        if ltp and ltp > 0:
            synth = _make_synth_df(
                float(ltp), n=int(os.getenv("SYNTH_WARMUP_BARS", "30"))
            )
            if synth is not None:
                try:
                    self._synth_bars_n = len(synth)
                except Exception:
                    pass
                log.warning(
                    "synthetic_warmup: using %d synthetic bars at LTP=%.2f (broker OHLC unavailable)",
                    len(synth),
                    float(ltp),
                )
                return synth

        try:
            import pandas as pd

            return pd.DataFrame()
        except Exception:
            return []

    cls._fetch_ohlc_df = wrap
    log.info("synthetic_warmup: patched LiveKiteSource._fetch_ohlc_df")


def _patch_have_min_bars() -> None:
    try:
        from src.data import source as ds  # type: ignore
    except Exception:
        return
    cls = getattr(ds, "LiveKiteSource", None)
    if not cls:
        return
    if hasattr(cls, "have_min_bars"):
        return

    def have_min_bars(self, n: int) -> bool:
        try:
            bb = getattr(self, "bar_builder", None)
            if bb and hasattr(bb, "have_min_bars"):
                return bool(bb.have_min_bars(int(n)))
        except Exception:
            pass
        try:
            sb = int(getattr(self, "_synth_bars_n", 0))
            if sb >= int(n):
                return True
        except Exception:
            pass
        return False

    cls.have_min_bars = have_min_bars
    log.info("synthetic_warmup: injected LiveKiteSource.have_min_bars")


def apply() -> None:
    """Activate runtime patches for synthetic warm-up bars."""

    if str(os.getenv("SYNTH_WARMUP_DISABLE", "false")).lower() in {
        "1",
        "true",
        "yes",
    }:
        return
    _patch_fetch_ohlc_df()
    _patch_have_min_bars()


try:  # pragma: no cover - defensive
    apply()
except Exception:  # pragma: no cover - defensive
    log.exception("synthetic_warmup: apply failed")

