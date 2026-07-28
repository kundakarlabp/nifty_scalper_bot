from pathlib import Path
import re

manager_path = Path("src/nifty_scalper_bot/core/strategy_manager.py")
text = manager_path.read_text()

text = text.replace(
    "        previous_snapshot = dict(self._latest_context_snapshots.get(role, {}) or {})\n",
    "",
    1,
)

pattern = re.compile(
    r"        derived_direction, derived_confidence, derived_reasons = self\._derive_context_direction\(indicators, role=role\)\n"
    r"        context_kind = \(\n"
    r".*?"
    r"        snapshot = \{\n",
    re.DOTALL,
)
replacement = '''        context_kind = (
            "price_direction"
            if role == "spot_context"
            else "volume_flow"
            if role == "futures_context"
            else "unknown"
        )
        current_volume = _num("volume")
        current_avg_volume = _num("avg_volume")
        futures_volume_ratio = _num("futures_volume_ratio")
        futures_volume_ratio_source = "indicator" if futures_volume_ratio is not None else "unavailable"
        if (
            role == "futures_context"
            and futures_volume_ratio is None
            and current_volume is not None
            and current_avg_volume is not None
            and current_avg_volume > 0
        ):
            futures_volume_ratio = current_volume / current_avg_volume
            futures_volume_ratio_source = "derived_volume_avg"
        vwap_slope = _num("vwap_slope")
        ema_slope = _num("ema_slope")
        vwap_slope_source = "indicator" if vwap_slope is not None else "unavailable"
        direction_inputs = dict(indicators)
        direction_inputs.update(
            futures_volume_ratio=futures_volume_ratio,
            vwap_slope=vwap_slope,
            ema_slope=ema_slope,
        )
        derived_direction, derived_confidence, derived_reasons = self._derive_context_direction(
            direction_inputs,
            role=role,
        )
        source_age_seconds = resolve_tick_age_seconds(indicators)
        if source_age_seconds is None:
            for source in (
                getattr(self, "_market_data_manager", None),
                getattr(self, "_data_hub", None),
            ):
                age_getter = getattr(source, "time_since_last_tick", None)
                if not callable(age_getter):
                    continue
                try:
                    candidate_age = age_getter(symbol)
                    source_age_seconds = None if candidate_age is None else max(0.0, float(candidate_age))
                except (TypeError, ValueError, RuntimeError):
                    source_age_seconds = None
                if source_age_seconds is not None:
                    break
        source_timestamp = indicators.get("timestamp") or indicators.get("context_timestamp_epoch") or 0.0
        if not source_timestamp and source_age_seconds is not None:
            source_timestamp = time.time() - source_age_seconds
        snapshot = {
'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit(f"strategy_manager context block substitutions={count}")

text = text.replace(
    '            "symbol": symbol, "role": role, "context_kind": context_kind, "timestamp": time.time(),\n',
    '            "symbol": symbol, "role": role, "context_kind": context_kind,\n            "timestamp": source_timestamp, "snapshot_updated_at": time.time(),\n            "tick_age_s": source_age_seconds,\n',
    1,
)
text = text.replace(
    '            "context_timestamp_epoch": time.time(),\n',
    '            "context_timestamp_epoch": source_timestamp,\n',
    1,
)

replacements = {
    '        vwap_slope = _f("vwap_slope") or 0.0\n': '        vwap_slope = _f("vwap_slope")\n',
    '        ema_slope = _f("ema_slope") or 0.0\n': '        ema_slope = _f("ema_slope")\n',
    '        futures_volume_ratio = _f("futures_volume_ratio") or 0.0\n': '        futures_volume_ratio = _f("futures_volume_ratio")\n',
    '            if close >= vwap: ce_score += 1.0; reasons.append("close_above_vwap")\n            else: pe_score += 1.0; reasons.append("close_below_vwap")\n': '            if close > vwap: ce_score += 1.0; reasons.append("close_above_vwap")\n            elif close < vwap: pe_score += 1.0; reasons.append("close_below_vwap")\n',
    '            if ema_fast >= ema_slow: ce_score += 1.0; reasons.append("ema_fast_above_slow")\n            else: pe_score += 1.0; reasons.append("ema_fast_below_slow")\n': '            if ema_fast > ema_slow: ce_score += 1.0; reasons.append("ema_fast_above_slow")\n            elif ema_fast < ema_slow: pe_score += 1.0; reasons.append("ema_fast_below_slow")\n',
    '            if close >= ema_50: ce_score += 0.5; reasons.append("close_above_ema50")\n            else: pe_score += 0.5; reasons.append("close_below_ema50")\n': '            if close > ema_50: ce_score += 0.5; reasons.append("close_above_ema50")\n            elif close < ema_50: pe_score += 0.5; reasons.append("close_below_ema50")\n',
    '        if vwap_slope > 0: ce_score += 0.5; reasons.append("vwap_slope_positive")\n        elif vwap_slope < 0: pe_score += 0.5; reasons.append("vwap_slope_negative")\n': '        if vwap_slope is not None and vwap_slope > 0: ce_score += 0.5; reasons.append("vwap_slope_positive")\n        elif vwap_slope is not None and vwap_slope < 0: pe_score += 0.5; reasons.append("vwap_slope_negative")\n',
    '        if ema_slope > 0: ce_score += 0.5; reasons.append("ema_slope_positive")\n        elif ema_slope < 0: pe_score += 0.5; reasons.append("ema_slope_negative")\n': '        if ema_slope is not None and ema_slope > 0: ce_score += 0.5; reasons.append("ema_slope_positive")\n        elif ema_slope is not None and ema_slope < 0: pe_score += 0.5; reasons.append("ema_slope_negative")\n',
    '        if role == "futures_context" and futures_volume_ratio >= 1.0:\n': '        if role == "futures_context" and futures_volume_ratio is not None and futures_volume_ratio >= 1.0:\n',
    '            if fut_usable:\n                indicators.setdefault("futures_context", fut_ctx)\n': '            if fut_fresh:\n                indicators["futures_context"] = fut_ctx\n',
    '                indicators.setdefault("futures_volume_ratio", fut_ctx.get("futures_volume_ratio"))\n': '                indicators["futures_volume_ratio"] = fut_ctx.get("futures_volume_ratio")\n',
    '                indicators.setdefault("futures_vwap", fut_ctx.get("vwap"))\n': '                indicators["futures_vwap"] = fut_ctx.get("vwap")\n',
    '                indicators.setdefault("futures_vwap_slope", fut_ctx.get("vwap_slope"))\n': '                indicators["futures_vwap_slope"] = fut_ctx.get("vwap_slope")\n',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"missing strategy_manager anchor: {old[:80]!r}")
    text = text.replace(old, new, 1)

old_age = '                indicators["context_age_seconds"] = min(now_ts - float(spot_ctx.get("timestamp", now_ts)) if spot_usable else max_context_age + 1, now_ts - float(fut_ctx.get("timestamp", now_ts)) if fut_usable else max_context_age + 1)\n'
new_age = '''                usable_context_ages = [
                    age
                    for age, usable in (
                        (spot_tick_age_s, spot_usable),
                        (fut_tick_age_s, fut_usable),
                    )
                    if usable and age is not None
                ]
                indicators["context_age_seconds"] = min(usable_context_ages) if usable_context_ages else max_context_age + 1
'''
if old_age not in text:
    raise SystemExit("missing context age anchor")
text = text.replace(old_age, new_age, 1)
manager_path.write_text(text)

vwap_path = Path("src/nifty_scalper_bot/strategies/elite_strategies/vwap_pro.py")
text = vwap_path.read_text()
old_parse = '''            futures_vwap_slope = float(indicators.get('futures_vwap_slope') or 0.0)
            futures_volume_ratio = float(indicators.get('futures_volume_ratio') or 0.0)
'''
new_parse = '''            def _optional_float(value: Any) -> float | None:
                try:
                    return None if value is None else float(value)
                except (TypeError, ValueError):
                    return None

            futures_vwap_slope = _optional_float(indicators.get('futures_vwap_slope'))
            futures_volume_ratio = _optional_float(indicators.get('futures_volume_ratio'))
'''
if old_parse not in text:
    raise SystemExit("VWAPPro parse anchor missing")
text = text.replace(old_parse, new_parse, 1)
old_volume = "            fut_vol_support = (contract_side == 'CE' and futures_volume_ratio >= 1.0) or (contract_side == 'PE' and futures_volume_ratio >= 1.0)\n"
if old_volume not in text:
    raise SystemExit("VWAPPro volume anchor missing")
text = text.replace(old_volume, "            fut_vol_support = futures_volume_ratio is not None and futures_volume_ratio >= 1.0\n", 1)
old_slope = '''            slope_available = abs(futures_vwap_slope) > self._futures_slope_neutral_eps
            if slope_available:
                slope_support = (contract_side == 'CE' and futures_vwap_slope > 0) or (contract_side == 'PE' and futures_vwap_slope < 0)
                if slope_support:
                    score += 1.0
                    reasons.append('futures_slope_alignment')
                else:
                    reasons.append('futures_slope_conflict')
            else:
                slope_support = False
                reasons.append('futures_slope_neutral')
'''
new_slope = '''            if futures_vwap_slope is None:
                slope_support = False
                reasons.append('futures_slope_unavailable')
            elif abs(futures_vwap_slope) <= self._futures_slope_neutral_eps:
                slope_support = False
                reasons.append('futures_slope_neutral')
            else:
                slope_support = (contract_side == 'CE' and futures_vwap_slope > 0) or (contract_side == 'PE' and futures_vwap_slope < 0)
                if slope_support:
                    score += 1.0
                    reasons.append('futures_slope_alignment')
                else:
                    reasons.append('futures_slope_conflict')
'''
if old_slope not in text:
    raise SystemExit("VWAPPro slope anchor missing")
text = text.replace(old_slope, new_slope, 1)
vwap_path.write_text(text)

candidates = sorted(Path("tests").rglob("test_strategy_manager*.py"))
if not candidates:
    raise SystemExit("no existing strategy manager test file found")
test_path = next(
    (path for path in candidates if "context" in path.name and "option" in path.name),
    candidates[0],
)
test_text = test_path.read_text()
if "test_futures_context_preserves_source_age_and_neutrality" not in test_text:
    test_text += '''


def test_futures_context_preserves_source_age_and_neutrality() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={
            "close": 25000.0,
            "vwap": 25000.0,
            "ema_fast": 25000.0,
            "ema_slow": 25000.0,
            "ema_50": 25000.0,
            "vwap_slope": 0.0,
            "ema_slope": 0.0,
            "tick_age_s": 180.0,
        },
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["tick_age_s"] == 180.0
    assert snapshot["timestamp"] < snapshot["snapshot_updated_at"]
    assert snapshot["vwap_slope"] == 0.0
    assert snapshot["direction_bias"] is None


def test_futures_context_uses_same_evaluation_slope_and_never_snapshot_delta() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyManager

    manager = object.__new__(StrategyManager)
    manager._latest_context_snapshots = {"futures_context": {"vwap": 100.0, "close": 100.0}}
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 101.0, "vwap": 101.0, "vwap_slope": 0.001, "tick_age_s": 0.1},
        role="futures_context",
    )
    snapshot = manager._latest_context_snapshots["futures_context"]
    assert snapshot["vwap_slope"] == 0.001
    assert snapshot["direction_bias"] == "CE"
    assert "vwap_slope_positive" in snapshot["direction_context_reasons"]
    manager._update_context_snapshot(
        symbol="NFO:NIFTY26JULFUT",
        indicators={"close": 102.0, "vwap": 102.0, "tick_age_s": 0.1},
        role="futures_context",
    )
    assert manager._latest_context_snapshots["futures_context"]["vwap_slope"] is None
'''
    test_path.write_text(test_text)
print(f"updated tests in {test_path}")
