#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(path: str, old: str, new: str, *, sentinel: str | None = None) -> None:
    value = read(path)
    if sentinel and sentinel in value:
        return
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}: {old[:100]!r}")
    write(path, value.replace(old, new, 1))


def insert_before(path: str, marker: str, block: str, *, sentinel: str) -> None:
    value = read(path)
    if sentinel in value:
        return
    count = value.count(marker)
    if count != 1:
        raise RuntimeError(f"{path}: expected one marker, found {count}: {marker!r}")
    write(path, value.replace(marker, block + marker, 1))


def patch_orderflow() -> None:
    path = "src/nifty_scalper_bot/strategies/elite_strategies/order_flow.py"
    replace_once(path, "import os\n", "import os\nimport time\n")
    replace_once(
        path,
        "from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy\n",
        "from nifty_scalper_bot.execution.quote_readiness import evaluate_execution_quote\nfrom nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteSignal, EliteStrategy\n",
    )
    replace_once(
        path,
        "        self._cfg = config\n",
        "        self._cfg = config\n        self._reversal_confirmation: dict[str, dict[str, Any]] = {}\n",
        sentinel="self._reversal_confirmation",
    )
    insert_before(
        path,
        "    def _evaluate_signal(self, symbol: str, indicators: dict[str, Any], current_price: float, position: Any | None = None) -> EliteSignal | None:\n",
        '''    def _reversal_persistence_confirmed(
        self,
        *,
        symbol: str,
        side: str,
        update_version: object | None,
        fingerprint: tuple[object, ...],
    ) -> bool:
        """Require distinct persistent updates before overriding direction bias."""
        version = update_version if update_version is not None else fingerprint
        now = time.monotonic()
        state = self._reversal_confirmation.get(symbol)
        if state is None or state.get("side") != side:
            state = {"side": side, "count": 1, "started": now, "version": version}
            self._reversal_confirmation[symbol] = state
        elif state.get("version") != version:
            state["count"] = int(state.get("count") or 0) + 1
            state["version"] = version
        try:
            min_updates = max(2, int(os.getenv("ORDERFLOW_REVERSAL_MIN_UPDATES", "3") or 3))
        except (TypeError, ValueError):
            min_updates = 3
        min_persistence_ms = max(
            0.0,
            safe_float_env("ORDERFLOW_REVERSAL_MIN_PERSISTENCE_MS", 500.0),
        )
        elapsed_ms = max(0.0, now - float(state.get("started") or now)) * 1000.0
        return int(state.get("count") or 0) >= min_updates and elapsed_ms >= min_persistence_ms

''',
        sentinel="def _reversal_persistence_confirmed",
    )
    replace_once(
        path,
        "            tradable_quote = bool(indicators.get('tradable_quote', True))\n            quote_depth_valid = bool(indicators.get('quote_depth_valid', True))\n            tick_age_ms = float(indicators.get('tick_age_ms') or 0.0)\n            max_tick_age_ms = float(os.getenv('LIVE_MAX_TICK_AGE_MS', '2500') or '2500')\n",
        "            tradable_quote = False\n            quote_depth_valid = False\n            tick_age_ms: float | None = None\n            quote_update_version: object | None = None\n            max_tick_age_ms = float(os.getenv('LIVE_MAX_TICK_AGE_MS', '2500') or '2500')\n",
        sentinel="quote_update_version: object | None",
    )
    insert_before(
        path,
        "            if total_bid + total_ask <= 0:\n",
        '''            quote_payload = dict(indicators)
            quote_payload.update(
                {
                    "bid": bid,
                    "ask": ask,
                    "depth": depth,
                    "depth_available": depth_available,
                    "spread_pct": spread_pct,
                }
            )
            quote_readiness = evaluate_execution_quote(
                symbol,
                quote_payload,
                live_mode=is_live_mode,
                max_tick_age_ms=max_tick_age_ms,
                max_spread_pct=trigger_max_spread_pct,
                require_depth=is_live_mode,
            )
            tradable_quote = quote_readiness.tradable_quote
            quote_depth_valid = quote_readiness.depth_available
            tick_age_ms = quote_readiness.tick_age_ms
            quote_update_version = quote_readiness.quote_update_version

''',
        sentinel="quote_readiness = evaluate_execution_quote",
    )
    replace_once(
        path,
        "            microstructure_confirms_side = bool(tick_supports and depth_available and imbalance_confirms)\n            bias_invalidated_by_microstructure = bool(bias_conflict and microstructure_confirms_side)\n            side_alignment_ok = (direction not in {'CE', 'PE'}) or side_aligns or bias_invalidated_by_microstructure\n",
        "            raw_microstructure_confirms_side = bool(tick_supports and depth_available and imbalance_confirms)\n            reversal_persistence_confirmed = False\n            if bias_conflict and raw_microstructure_confirms_side:\n                reversal_persistence_confirmed = (\n                    not is_live_mode\n                    or self._reversal_persistence_confirmed(\n                        symbol=symbol,\n                        side=side,\n                        update_version=quote_update_version,\n                        fingerprint=(tick_age_ms, round(bid, 4), round(ask, 4), round(total_bid, 2), round(total_ask, 2)),\n                    )\n                )\n            else:\n                self._reversal_confirmation.pop(symbol, None)\n            microstructure_confirms_side = bool(\n                raw_microstructure_confirms_side\n                and (not is_live_mode or reversal_persistence_confirmed)\n            )\n            bias_invalidated_by_microstructure = bool(bias_conflict and microstructure_confirms_side)\n            side_alignment_ok = (direction not in {'CE', 'PE'}) or side_aligns or bias_invalidated_by_microstructure\n",
        sentinel="raw_microstructure_confirms_side",
    )
    duplicate = '''            if is_live_mode and not selected_meta_available:
                selected_or_near_atm = False
            if bias_invalidated_by_microstructure:
                LOGGER.info(
                    'ORDERFLOW_STALE_BIAS_INVALIDATED symbol=%s side=%s stale_bias=%s '
                    'depth_imbalance=%.3f tick_direction=%s score=%.2f',
                    symbol, side, direction, depth_imbalance, tick_direction, strategy_score,
                    extra={
                        'event': 'ORDERFLOW_STALE_BIAS_INVALIDATED',
                        'symbol': symbol, 'side': side, 'stale_bias': direction,
                        'depth_imbalance': round(depth_imbalance, 4),
                        'tick_direction': tick_direction, 'score': strategy_score,
                    },
                )
'''
    replace_once(
        path,
        duplicate,
        "            if is_live_mode and not selected_meta_available:\n                selected_or_near_atm = False\n",
    )
    replace_once(
        path,
        "                and tick_supports\n                and tick_age_ms <= max_tick_age_ms\n            )\n",
        "                and tick_supports\n                and quote_readiness.allowed\n                and tick_age_ms is not None\n                and tick_age_ms <= max_tick_age_ms\n            )\n",
        sentinel="and quote_readiness.allowed",
    )
    replace_once(
        path,
        "            conflict_override_requested = bool(\n                not side_alignment_ok\n",
        "            conflict_override_requested = bool(\n                not is_live_mode\n                and not side_alignment_ok\n",
        sentinel="not is_live_mode\n                and not side_alignment_ok",
    )
    replace_once(
        path,
        "                and bool(tick_supports)\n                and tick_age_ms <= max_tick_age_ms\n                and bool(selected_or_near_atm)\n",
        "                and bool(tick_supports)\n                and tick_age_ms is not None\n                and tick_age_ms <= max_tick_age_ms\n                and bool(selected_or_near_atm)\n",
    )
    replace_once(
        path,
        "            elif not quote_depth_valid or not depth_available:\n                trigger_block_reason = 'quote_depth_missing'\n",
        "            elif not quote_readiness.allowed:\n                trigger_block_reason = quote_readiness.reason\n            elif not quote_depth_valid or not depth_available:\n                trigger_block_reason = 'quote_depth_missing'\n",
        sentinel="trigger_block_reason = quote_readiness.reason",
    )
    replace_once(path, "            elif tick_age_ms > max_tick_age_ms:\n", "            elif tick_age_ms is None or tick_age_ms > max_tick_age_ms:\n")
    value = read(path)
    old = "'quote_update_version': indicators.get('quote_update_version')"
    if value.count(old) != 2:
        raise RuntimeError(f"{path}: expected two quote_update_version metadata anchors")
    value = value.replace(old, "'quote_update_version': quote_update_version", 2)
    value = value.replace(
        "                 'selected_or_near_atm': selected_or_near_atm,\n",
        "                 'quote_readiness_allowed': quote_readiness.allowed,\n                 'quote_readiness_reason': quote_readiness.reason,\n                 'real_ticks_last_60s': quote_readiness.real_ticks_last_60s,\n                 'real_tick_count_derived': quote_readiness.real_tick_count_derived,\n                 'reversal_persistence_confirmed': reversal_persistence_confirmed,\n                 'selected_or_near_atm': selected_or_near_atm,\n",
        1,
    )
    write(path, value)


def patch_selector() -> None:
    path = "src/nifty_scalper_bot/strategies/trade_selector.py"
    replace_once(
        path,
        "from nifty_scalper_bot.config.env_utils import parse_int_env\n",
        "from nifty_scalper_bot.config.env_utils import parse_int_env\nfrom nifty_scalper_bot.execution.quote_readiness import (\n    resolve_real_tick_count,\n    resolve_tick_age_ms,\n)\n",
    )
    replace_once(
        path,
        "            tick_age_s = self._f(s.get('tick_age_s'))\n            if tick_age_s is None or tick_age_s > max_age:\n                rejects['tick_stale'] += 1\n                self._log_reject(\"tick_stale\", symbol, throttle_key_parts=(\"tick_stale\", symbol, int(max_age)), tick_age_s=tick_age_s, max_age_s=max_age)\n                continue\n            real_ticks = int(s.get('real_ticks_last_60s') or 0)\n            if real_ticks < min_ticks:\n                rejects['insufficient_ticks'] += 1\n                self._log_reject(\"insufficient_ticks\", symbol, throttle_key_parts=(\"insufficient_ticks\", symbol, min_ticks), real_ticks_last_60s=real_ticks, min_ticks=min_ticks)\n                continue\n            reasons = ['candidate_valid']\n",
        "            tick_age_ms = resolve_tick_age_ms(s)\n            tick_age_s = None if tick_age_ms is None else tick_age_ms / 1000.0\n            if tick_age_s is None or tick_age_s > max_age:\n                rejects['tick_stale'] += 1\n                self._log_reject(\"tick_stale\", symbol, throttle_key_parts=(\"tick_stale\", symbol, int(max_age)), tick_age_s=tick_age_s, tick_age_ms=tick_age_ms, max_age_s=max_age)\n                continue\n            real_ticks, real_ticks_derived = resolve_real_tick_count(\n                s, tick_age_ms=tick_age_ms, max_age_ms=max_age * 1000.0, has_bid_ask=has_bid_ask\n            )\n            if real_ticks < min_ticks:\n                rejects['insufficient_ticks'] += 1\n                self._log_reject(\"insufficient_ticks\", symbol, throttle_key_parts=(\"insufficient_ticks\", symbol, min_ticks), real_ticks_last_60s=real_ticks, real_tick_count_derived=real_ticks_derived, min_ticks=min_ticks)\n                continue\n            reasons = ['candidate_valid']\n            if real_ticks_derived:\n                reasons.append('real_tick_count_derived_from_fresh_ms_quote')\n",
        sentinel="real_tick_count_derived_from_fresh_ms_quote",
    )


if __name__ == "__main__":
    patch_orderflow()
    patch_selector()
