from __future__ import annotations

import ast
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def replace(path: str, old: str, new: str) -> None:
    text = read(path)
    if old not in text:
        raise RuntimeError(f"missing anchor in {path}: {old[:120]!r}")
    write(path, text.replace(old, new, 1))


# 1. Conservative cost estimate when bid/ask is unavailable.
replace(
    "src/nifty_scalper_bot/strategies/trade_selector.py",
    "            half_spread = (((ask or 0.0) - (bid or 0.0)) / 2.0) if has_bid_ask else entry * 0.003\n",
    "            ltp_only_half_spread_pct = max(\n"
    "                0.0,\n"
    "                float(os.getenv('LTP_ONLY_HALF_SPREAD_PCT', '0.02') or 0.02),\n"
    "            )\n"
    "            half_spread = (\n"
    "                ((ask or 0.0) - (bid or 0.0)) / 2.0\n"
    "                if has_bid_ask\n"
    "                else entry * ltp_only_half_spread_pct\n"
    "            )\n",
)

# 2. Cumulative current-session VWAP and zero-volume SMA fallback.
VWAP = "src/nifty_scalper_bot/strategies/vwap_mean_reversion.py"
replace(
    VWAP,
    "from dataclasses import dataclass\nfrom typing import Any, Literal, Mapping, Sequence\n",
    "from dataclasses import dataclass\nfrom datetime import date, datetime, timezone\nfrom typing import Any, Literal, Mapping, Sequence\n",
)
replace(
    VWAP,
    "\n\n@dataclass(slots=True, frozen=True)\nclass VWAPSignal:\n",
    '''\n\ndef _bar_session_date(bar: Mapping[str, Any]) -> date | None:\n    """Return the trading date encoded by a bar timestamp, when available."""\n    raw = next(\n        (bar.get(key) for key in ("timestamp", "datetime", "date", "time") if bar.get(key) is not None),\n        None,\n    )\n    if isinstance(raw, datetime):\n        return raw.date()\n    if isinstance(raw, date):\n        return raw\n    if isinstance(raw, (int, float)):\n        try:\n            return datetime.fromtimestamp(float(raw), tz=timezone.utc).date()\n        except (OSError, OverflowError, ValueError):\n            return None\n    if isinstance(raw, str):\n        value = raw.strip()\n        if not value:\n            return None\n        try:\n            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()\n        except ValueError:\n            try:\n                return date.fromisoformat(value[:10])\n            except ValueError:\n                return None\n    return None\n\n\n@dataclass(slots=True, frozen=True)\nclass VWAPSignal:\n''',
)
old_block = '''            bars = list(bars_payload)[-self._lookback :]\n            closes: list[float] = []\n            volumes: list[float] = []\n            for bar in bars:\n                if not isinstance(bar, Mapping):\n                    continue\n                close_value = _coerce_float(bar.get("close"))\n                volume_value = _coerce_float(bar.get("volume"))\n                if close_value is None or volume_value is None:\n                    continue\n                closes.append(close_value)\n                volumes.append(max(volume_value, 0.0))\n            total_volume = sum(volumes)\n            if total_volume <= 0.0 or not closes:\n                logger.info(\n                    'Condition met: volume_data_missing',\n                    extra={'event': 'vwap_mean_reversion_zero_volume'},\n                )\n                return VWAPSignal(\n                    action="HOLD",\n                    reason="volume_data_missing",\n                    metadata={"threshold_pct": self._threshold_pct},\n                )\n            vwap_numerator = sum(\n                close * volume for close, volume in zip(closes, volumes)\n            )\n            vwap = vwap_numerator / total_volume\n'''
new_block = '''            raw_bars = [bar for bar in bars_payload if isinstance(bar, Mapping)]\n            dated_bars = [(_bar_session_date(bar), bar) for bar in raw_bars]\n            session_dates = [session_date for session_date, _ in dated_bars if session_date]\n            latest_session = max(session_dates) if session_dates else None\n            bars = (\n                [bar for session_date, bar in dated_bars if session_date == latest_session]\n                if latest_session is not None\n                else raw_bars\n            )\n            closes: list[float] = []\n            volumes: list[float] = []\n            for bar in bars:\n                close_value = _coerce_float(bar.get("close"))\n                if close_value is None or close_value <= 0:\n                    continue\n                volume_value = _coerce_float(bar.get("volume"))\n                closes.append(close_value)\n                volumes.append(max(volume_value or 0.0, 0.0))\n            if not closes:\n                logger.info(\n                    'Condition met: price_data_missing',\n                    extra={'event': 'vwap_mean_reversion_no_valid_closes'},\n                )\n                return VWAPSignal(\n                    action="HOLD",\n                    reason="price_data_missing",\n                    metadata={"threshold_pct": self._threshold_pct},\n                )\n            total_volume = sum(volumes)\n            if total_volume > 0.0:\n                vwap_numerator = sum(\n                    close * volume for close, volume in zip(closes, volumes)\n                )\n                vwap = vwap_numerator / total_volume\n                benchmark_source = "session_vwap"\n            else:\n                vwap = sum(closes) / len(closes)\n                benchmark_source = "session_sma_zero_volume_fallback"\n                logger.warning(\n                    'VWAP volume unavailable; using session SMA fallback',\n                    extra={\n                        'event': 'vwap_mean_reversion_sma_fallback',\n                        'bars_used': len(closes),\n                    },\n                )\n'''
replace(VWAP, old_block, new_block)
replace(
    VWAP,
    "                'bars_used': len(closes),\n                'threshold_pct': threshold,\n",
    "                'bars_used': len(closes),\n"
    "                'session_date': latest_session.isoformat() if latest_session else None,\n"
    "                'benchmark_source': benchmark_source,\n"
    "                'threshold_pct': threshold,\n",
)
replace(
    VWAP,
    "            if len(closes) > 1 and vwap > 0:\n                mean_close = sum(closes) / len(closes)\n                variance = sum(\n                    (close - mean_close) ** 2 for close in closes\n                ) / len(closes)\n",
    "            volatility_closes = closes[-self._lookback :]\n"
    "            if len(volatility_closes) > 1 and vwap > 0:\n"
    "                mean_close = sum(volatility_closes) / len(volatility_closes)\n"
    "                variance = sum(\n"
    "                    (close - mean_close) ** 2 for close in volatility_closes\n"
    "                ) / len(volatility_closes)\n",
)

# 3. Explicit multi/all mode while retaining the expiry-gamma safety opt-in.
BUILDER = "src/nifty_scalper_bot/strategies/elite_strategies/builder.py"
replace(
    BUILDER,
    "    strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).strip().lower()\n",
    "    strategy_mode = str(os.getenv('STRATEGY_MODE', 'directional_scalp')).strip().lower()\n"
    "    multi_mode = strategy_mode in {'all', 'multi'}\n",
)
replace(BUILDER, "            if strategy_mode == 'directional_scalp':\n", "            if strategy_mode == 'directional_scalp' and not multi_mode:\n")
replace(
    BUILDER,
    "                strategy_mode == 'expiry_gamma' and allow_expiry_gamma\n",
    "                (strategy_mode == 'expiry_gamma' or multi_mode) and allow_expiry_gamma\n",
)
replace(
    BUILDER,
    "            if field_name in _THETA_ONLY and strategy_mode != 'theta':\n",
    "            if field_name in _THETA_ONLY and strategy_mode != 'theta' and not multi_mode:\n",
)

for path, old, new in (
    (
        "src/nifty_scalper_bot/strategies/elite_strategies/gamma_scalping.py",
        "            if not (strategy_mode == 'expiry_gamma' and gamma_enabled):\n",
        "            if not (strategy_mode in {'expiry_gamma', 'all', 'multi'} and gamma_enabled):\n",
    ),
    (
        "src/nifty_scalper_bot/strategies/elite_tuesday_gamma_buyer.py",
        "            if not (strategy_mode == 'expiry_gamma' and gamma_enabled):\n",
        "            if not (strategy_mode in {'expiry_gamma', 'all', 'multi'} and gamma_enabled):\n",
    ),
    (
        "src/nifty_scalper_bot/strategies/elite_strategies/straddle_theta.py",
        "            if str(os.getenv('STRATEGY_MODE', 'directional_scalp')).lower() != 'theta':\n",
        "            if str(os.getenv('STRATEGY_MODE', 'directional_scalp')).lower() not in {'theta', 'all', 'multi'}:\n",
    ),
):
    replace(path, old, new)

# 4. Market-order slippage reference, resolved after submission to avoid delaying execution.
ORDER = "src/nifty_scalper_bot/execution/order_manager_core.py"
replace(
    ORDER,
    "    rejection_reason: str | None = None\n",
    "    rejection_reason: str | None = None\n"
    "    reference_price: float | None = None\n",
)
helper = '''\n    def _market_order_reference_price(\n        self,\n        symbol: str,\n        response: Mapping[str, Any],\n    ) -> float | None:\n        """Resolve a near-submit market reference without delaying order placement."""\n        for key in ("reference_price", "last_price", "ltp", "close"):\n            value = response.get(key)\n            try:\n                numeric = float(value)\n            except (TypeError, ValueError):\n                continue\n            if math.isfinite(numeric) and numeric > 0:\n                return numeric\n        broker = getattr(self, "_broker", None)\n        getter = getattr(broker, "get_quote", None)\n        if not callable(getter):\n            return None\n        try:\n            quote = getter(symbol)\n        except Exception as exc:  # noqa: BLE001 - slippage telemetry is non-blocking\n            self._logger.debug(\n                "market_reference_quote_failed",\n                extra={\n                    "event": "market_reference_quote_failed",\n                    "symbol": symbol,\n                    "error": str(exc),\n                },\n            )\n            return None\n        if isinstance(quote, Mapping) and symbol in quote and isinstance(quote[symbol], Mapping):\n            quote = quote[symbol]\n        if not isinstance(quote, Mapping):\n            return None\n        for key in ("last_price", "ltp", "close"):\n            try:\n                numeric = float(quote.get(key))\n            except (TypeError, ValueError):\n                continue\n            if math.isfinite(numeric) and numeric > 0:\n                return numeric\n        return None\n\n'''
replace(ORDER, "    def _place_single_order(\n", helper + "    def _place_single_order(\n")
replace(
    ORDER,
    "        resolved_client_id = str(response_client_id) if response_client_id else None\n        details = OrderDetails(\n",
    "        resolved_client_id = str(response_client_id) if response_client_id else None\n"
    "        reference_price = (\n"
    "            self._market_order_reference_price(symbol, response)\n"
    "            if order_type == OrderType.MARKET and not price\n"
    "            else None\n"
    "        )\n"
    "        details = OrderDetails(\n",
)
replace(
    ORDER,
    "            price=float(price or 0.0),\n            status=status,\n",
    "            price=float(price or reference_price or 0.0),\n"
    "            reference_price=reference_price,\n"
    "            status=status,\n",
)

# 5. Persisted position time-stop, shorter only when weekly symbol proves expiry today.
BRACKET = "src/nifty_scalper_bot/execution/bracket_core.py"
replace(BRACKET, "import os\n", "import os\nimport re\n")
replace(
    BRACKET,
    "    entry_fill_price: float | None = None\n",
    "    entry_fill_price: float | None = None\n"
    "    entry_filled_at: float | None = None\n",
)
replace(
    BRACKET,
    '            "entry_fill_price": self.entry_fill_price,\n',
    '            "entry_fill_price": self.entry_fill_price,\n'
    '            "entry_filled_at": self.entry_filled_at,\n',
)
replace(
    BRACKET,
    "            bracket.entry_fill_price = fill_price\n            bracket.exit_pending = False\n",
    "            bracket.entry_fill_price = fill_price\n"
    "            bracket.entry_filled_at = time.time()\n"
    "            bracket.exit_pending = False\n",
)
replace(
    BRACKET,
    "        entry_fill_price=payload.get(\"entry_fill_price\"),\n",
    "        entry_fill_price=payload.get(\"entry_fill_price\"),\n"
    "        entry_filled_at=payload.get(\"entry_filled_at\"),\n",
)
time_helpers = '''\n    @staticmethod\n    def _weekly_option_expiry_date(symbol: str) -> datetime | None:\n        """Parse NSE weekly YYMDD expiry encoding; return None when ambiguous."""\n        token = normalize_symbol(symbol).split(":")[-1].upper()\n        match = re.match(r"^NIFTY(?P<yy>\\d{2})(?P<month>[1-9OND])(?P<day>\\d{2})\\d+(?:CE|PE)$", token)\n        if not match:\n            return None\n        month_token = match.group("month")\n        month_map = {"O": 10, "N": 11, "D": 12}\n        month = month_map.get(month_token, int(month_token) if month_token.isdigit() else 0)\n        try:\n            return datetime(2000 + int(match.group("yy")), month, int(match.group("day")), tzinfo=timezone.utc)\n        except ValueError:\n            return None\n\n    def _position_time_stop_seconds(self, bracket: BracketState) -> float | None:\n        normal_minutes = parse_float_env(os.getenv("POSITION_TIME_STOP_MIN"), 12.0)\n        if normal_minutes <= 0:\n            return None\n        expiry_minutes = parse_float_env(os.getenv("EXPIRY_DAY_TIME_STOP_MIN"), 4.0)\n        expiry = self._weekly_option_expiry_date(bracket.symbol)\n        now_utc = datetime.now(timezone.utc)\n        minutes = (\n            expiry_minutes\n            if expiry is not None and expiry.date() == now_utc.date() and expiry_minutes > 0\n            else normal_minutes\n        )\n        return minutes * 60.0\n\n'''
replace(BRACKET, "    def _evaluate_exit_fast(self, bracket: BracketState, ltp: float) -> dict | None:\n", time_helpers + "    def _evaluate_exit_fast(self, bracket: BracketState, ltp: float) -> dict | None:\n")
replace(
    BRACKET,
    "        # Check partial targets first (TP1, TP2, etc.)\n",
    "        time_stop_seconds = self._position_time_stop_seconds(bracket)\n"
    "        filled_at = bracket.entry_filled_at\n"
    "        if filled_at is None and bracket.entry_confirmed:\n"
    "            filled_at = bracket.created_at\n"
    "        if (\n"
    "            time_stop_seconds is not None\n"
    "            and filled_at is not None\n"
    "            and time.time() - float(filled_at) >= time_stop_seconds\n"
    "        ):\n"
    "            return {\n"
    "                \"type\": \"TIME_STOP\",\n"
    "                \"price\": ltp,\n"
    "                \"qty\": bracket.remaining_quantity,\n"
    "                \"reason\": f\"POSITION_TIME_STOP_{int(time_stop_seconds // 60)}M\",\n"
    "            }\n"
    "\n"
    "        # Check partial targets first (TP1, TP2, etc.)\n",
)

# Tests.
write(
    "tests/strategies/test_market_aware_optimisations.py",
    '''from __future__ import annotations\n\nfrom datetime import datetime, timezone\n\nimport pytest\n\nfrom nifty_scalper_bot.strategies.vwap_mean_reversion import VWAPMeanReversionStrategy\n\n\ndef test_vwap_uses_complete_latest_session_not_rolling_lookback() -> None:\n    bars = [\n        {"timestamp": "2026-06-28T09:15:00+05:30", "close": 100.0, "volume": 10},\n        {"timestamp": "2026-06-28T09:16:00+05:30", "close": 110.0, "volume": 10},\n        {"timestamp": "2026-06-29T09:15:00+05:30", "close": 200.0, "volume": 10},\n        {"timestamp": "2026-06-29T09:16:00+05:30", "close": 220.0, "volume": 30},\n    ]\n    signal = VWAPMeanReversionStrategy(lookback=1).generate_signal(\n        {"NIFTY": {"ltp": 210.0}, "NIFTYFUT_bars": bars}\n    )\n    assert signal.metadata["vwap"] == pytest.approx(215.0)\n    assert signal.metadata["bars_used"] == 2\n    assert signal.metadata["session_date"] == "2026-06-29"\n    assert signal.metadata["benchmark_source"] == "session_vwap"\n\n\ndef test_zero_volume_uses_session_sma_instead_of_failing() -> None:\n    signal = VWAPMeanReversionStrategy().generate_signal(\n        {\n            "NIFTY": {"ltp": 105.0},\n            "NIFTY_bars": [\n                {"timestamp": "2026-06-29T09:15:00+05:30", "close": 100.0, "volume": 0},\n                {"timestamp": "2026-06-29T09:16:00+05:30", "close": 110.0, "volume": 0},\n            ],\n        }\n    )\n    assert signal.reason != "volume_data_missing"\n    assert signal.metadata["vwap"] == pytest.approx(105.0)\n    assert signal.metadata["benchmark_source"] == "session_sma_zero_volume_fallback"\n''',
)
write(
    "tests/execution/test_market_aware_time_stop.py",
    '''from __future__ import annotations\n\nfrom datetime import datetime, timezone\nfrom types import SimpleNamespace\nimport time\n\nfrom nifty_scalper_bot.execution import BracketManager\nfrom nifty_scalper_bot.execution.bracket_core import BracketState\n\n\ndef _stop(manager: BracketManager) -> None:\n    manager.shutdown()\n    manager._watchdog_thread.join(timeout=1.0)\n\n\ndef test_normal_position_time_stop(monkeypatch, tmp_path) -> None:\n    monkeypatch.setenv("DATA_DIR", str(tmp_path))\n    monkeypatch.setenv("POSITION_TIME_STOP_MIN", "12")\n    manager = BracketManager(order_manager=SimpleNamespace())\n    _stop(manager)\n    bracket = BracketState(\n        entry_order_id="entry", symbol="NFO:NIFTY2663024000CE", side="BUY",\n        quantity=65, entry_price=100.0, sl_trigger_price=90.0, tp_trigger_price=120.0,\n        entry_confirmed=True, entry_filled_at=time.time() - 721,\n    )\n    action = manager._evaluate_exit_fast(bracket, 101.0)\n    assert action is not None\n    assert action["type"] == "TIME_STOP"\n    assert action["reason"] == "POSITION_TIME_STOP_12M"\n\n\ndef test_expiry_day_time_stop_is_shorter(monkeypatch, tmp_path) -> None:\n    monkeypatch.setenv("DATA_DIR", str(tmp_path))\n    monkeypatch.setenv("POSITION_TIME_STOP_MIN", "12")\n    monkeypatch.setenv("EXPIRY_DAY_TIME_STOP_MIN", "4")\n    now = datetime.now(timezone.utc)\n    month_token = str(now.month) if now.month <= 9 else {10: "O", 11: "N", 12: "D"}[now.month]\n    symbol = f"NFO:NIFTY{now.year % 100:02d}{month_token}{now.day:02d}24000CE"\n    manager = BracketManager(order_manager=SimpleNamespace())\n    _stop(manager)\n    bracket = BracketState(\n        entry_order_id="entry", symbol=symbol, side="BUY", quantity=65,\n        entry_price=100.0, sl_trigger_price=90.0, tp_trigger_price=120.0,\n        entry_confirmed=True, entry_filled_at=time.time() - 241,\n    )\n    action = manager._evaluate_exit_fast(bracket, 101.0)\n    assert action is not None\n    assert action["reason"] == "POSITION_TIME_STOP_4M"\n''',
)

for path in [
    "src/nifty_scalper_bot/strategies/trade_selector.py",
    VWAP,
    BUILDER,
    "src/nifty_scalper_bot/strategies/elite_strategies/gamma_scalping.py",
    "src/nifty_scalper_bot/strategies/elite_tuesday_gamma_buyer.py",
    "src/nifty_scalper_bot/strategies/elite_strategies/straddle_theta.py",
    ORDER,
    BRACKET,
    "tests/strategies/test_market_aware_optimisations.py",
    "tests/execution/test_market_aware_time_stop.py",
]:
    ast.parse(read(path), filename=path)

print("market-aware optimisations applied")
