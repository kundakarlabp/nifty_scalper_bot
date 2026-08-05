from pathlib import Path

path = Path("src/nifty_scalper_bot/core/strategy_manager.py")
text = path.read_text()
old_helper = '''        def _num(*keys: str) -> float | None:
            for key in keys:
                value = indicators.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        context_kind = (
'''
new_helper = '''        def _num(*keys: str) -> float | None:
            for key in keys:
                value = indicators.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        def _history_ema(period: int) -> float | None:
            if role != "futures_context":
                return None
            getter = getattr(getattr(self, "_indicator_engine", None), "get_ema", None)
            if not callable(getter):
                return None
            try:
                value = getter(symbol, period=period)
            except TypeError:
                value = getter(symbol, period)
            except Exception as exc:  # noqa: BLE001 - context remains fail-closed
                log.debug(
                    "FUTURES_CONTEXT_EMA_FALLBACK_FAILED symbol=%s period=%s error=%s",
                    symbol,
                    period,
                    exc,
                )
                return None
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        context_kind = (
'''
if old_helper not in text:
    raise SystemExit("context helper patch needle not found")
text = text.replace(old_helper, new_helper, 1)

old_metrics = '''        vwap_slope = _num("vwap_slope")
        ema_slope = _num("ema_slope")
        vwap_slope_source = "indicator" if vwap_slope is not None else "unavailable"
        direction_inputs = dict(indicators)
        direction_inputs.update(
            futures_volume_ratio=futures_volume_ratio,
            vwap_slope=vwap_slope,
            ema_slope=ema_slope,
        )
'''
new_metrics = '''        vwap_slope = _num("vwap_slope")
        ema_slope = _num("ema_slope")
        vwap_slope_source = "indicator" if vwap_slope is not None else "unavailable"
        ema_fast = _num("ema_fast", "ema_9", "ema9")
        ema_slow = _num("ema_slow", "ema_21", "ema21")
        ema_50 = _num("ema_50", "ema50")
        ema_fast_source = "indicator" if ema_fast is not None else "unavailable"
        ema_slow_source = "indicator" if ema_slow is not None else "unavailable"
        ema_50_source = "indicator" if ema_50 is not None else "unavailable"
        if role == "futures_context":
            if ema_fast is None:
                ema_fast = _history_ema(9)
                if ema_fast is not None:
                    ema_fast_source = "indicator_engine_history"
            if ema_slow is None:
                ema_slow = _history_ema(21)
                if ema_slow is not None:
                    ema_slow_source = "indicator_engine_history"
            if ema_50 is None:
                ema_50 = _history_ema(50)
                if ema_50 is not None:
                    ema_50_source = "indicator_engine_history"
        direction_inputs = dict(indicators)
        direction_inputs.update(
            futures_volume_ratio=futures_volume_ratio,
            vwap_slope=vwap_slope,
            ema_slope=ema_slope,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_50=ema_50,
        )
'''
if old_metrics not in text:
    raise SystemExit("context metrics patch needle not found")
text = text.replace(old_metrics, new_metrics, 1)

old_snapshot = '''            "vwap": _num("exchange_vwap", "session_vwap", "vwap"),
            "ema_fast": _num("ema_fast", "ema_9", "ema9"),
            "ema_slow": _num("ema_slow", "ema_21", "ema21"), "ema_50": _num("ema_50", "ema50"),
            "adx": _num("adx"), "atr": _num("atr"), "volume": _num("volume"),
'''
new_snapshot = '''            "vwap": _num("exchange_vwap", "session_vwap", "vwap"),
            "ema_fast": ema_fast,
            "ema_slow": ema_slow, "ema_50": ema_50,
            "ema_fast_source": ema_fast_source,
            "ema_slow_source": ema_slow_source,
            "ema_50_source": ema_50_source,
            "adx": _num("adx"), "atr": _num("atr"), "volume": _num("volume"),
'''
if old_snapshot not in text:
    raise SystemExit("context snapshot patch needle not found")
text = text.replace(old_snapshot, new_snapshot, 1)
path.write_text(text)

test_path = Path("tests/core/test_strategy_manager_context_age.py")
test_text = test_path.read_text()
regression = '''

def test_futures_context_uses_hydrated_history_when_ema_aliases_are_absent() -> None:
    from datetime import datetime, timedelta, timezone

    from nifty_scalper_bot.strategies.indicators import IndicatorEngine

    symbol = "NFO:NIFTY26AUGFUT"
    engine = IndicatorEngine()
    started_at = datetime(2026, 8, 5, 3, 45, tzinfo=timezone.utc)
    for index in range(60):
        price = 25000.0 + float(index)
        engine.update_price(
            symbol,
            {"open": price, "high": price, "low": price, "close": price},
            volume=0,
            timestamp=started_at + timedelta(minutes=index),
        )

    manager = object.__new__(StrategyManager)
    manager._indicator_engine = engine
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol=symbol,
        indicators={"ltp": 25059.0, "close": 25059.0},
        role="futures_context",
    )

    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap"] is None
    assert snapshot["ema_fast"] > snapshot["ema_slow"] > snapshot["ema_50"]
    assert snapshot["ema_fast_source"] == "indicator_engine_history"
    assert snapshot["ema_slow_source"] == "indicator_engine_history"
    assert snapshot["ema_50_source"] == "indicator_engine_history"
    assert snapshot["direction_bias"] == "CE"
    assert "ema_fast_above_slow" in snapshot["direction_context_reasons"]
'''
if "test_futures_context_uses_hydrated_history_when_ema_aliases_are_absent" in test_text:
    raise SystemExit("regression already exists")
test_path.write_text(test_text.rstrip() + regression + "\n")
