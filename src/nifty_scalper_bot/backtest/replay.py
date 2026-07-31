"""Minute-bar replay harness for exercising the live trading stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd

from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine


class ReplayClock(Protocol):
    """Clock surface used by deterministic historical replay."""

    def advance_to(self, timestamp: datetime) -> None: ...


@dataclass(slots=True)
class ReplayResult:
    """Outcome of a replay run for downstream reporting."""

    bars_processed: int
    start: datetime | None
    end: datetime | None
    trades: list[dict[str, Any]]
    orders: list[dict[str, Any]]

    def format_summary(self) -> str:
        """Return a human-readable summary suitable for Telegram."""

        if self.bars_processed == 0:
            return "Replay produced no bars."
        start = self.start.isoformat() if self.start else "?"
        end = self.end.isoformat() if self.end else "?"
        return (
            f"Bars: {self.bars_processed}\n"
            f"Window: {start} → {end}\n"
            f"Orders: {len(self.orders)} | Trades: {len(self.trades)}"
        )


class ReplayHarness:
    """Feed historical minute bars through :class:`StrategyRunner`."""

    def __init__(
        self,
        runner: Any,
        paper_engine: PaperFillEngine,
        *,
        option_symbol: str,
        index_symbol: str | None = None,
        clock: ReplayClock | None = None,
    ) -> None:
        self._runner = runner
        self._paper = paper_engine
        self._option_symbol = option_symbol
        self._index_symbol = index_symbol
        self._clock = clock
        clock_time = getattr(clock, "time", None)
        set_clock = getattr(paper_engine, "set_clock", None)
        if callable(clock_time) and callable(set_clock):
            set_clock(clock_time)

    def run_dataframe(self, data: pd.DataFrame) -> ReplayResult:
        """Replay the provided dataframe and return a :class:`ReplayResult`."""

        if data.empty:
            return ReplayResult(0, None, None, [], [])
        frame = data.copy()
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)
        start_ts = frame.index[0].to_pydatetime()
        end_ts = frame.index[-1].to_pydatetime()
        bars = 0
        for timestamp, row in frame.iterrows():
            tick_time = timestamp.to_pydatetime()
            if self._clock is not None:
                self._clock.advance_to(tick_time)
            if self._index_symbol:
                index_tick = _build_tick(row, tick_time, prefix="index_")
                if index_tick:
                    self._publish_and_dispatch(self._index_symbol, index_tick)
            option_tick = _build_tick(row, tick_time, prefix="option_")
            if option_tick:
                self._publish_and_dispatch(self._option_symbol, option_tick)
            bars += 1
        trades = _collect_trade_history(self._runner)
        orders = self._paper.get_orders() if hasattr(self._paper, "get_orders") else []
        return ReplayResult(bars, start_ts, end_ts, trades, orders)

    def run_file(self, path: Path) -> ReplayResult:
        """Load ``path`` as CSV/Parquet and run :meth:`run_dataframe`."""

        frame = _load_frame(path)
        return self.run_dataframe(frame)

    def run_day(self, directory: Path, day: str) -> ReplayResult:
        """Run replay for ``day`` located inside ``directory``."""

        day_path = _resolve_day_path(directory, day)
        return self.run_file(day_path)

    def _publish_and_dispatch(self, symbol: str, tick: Mapping[str, Any]) -> None:
        payload = {
            **dict(tick),
            "symbol": symbol,
            "last_price": float(tick["close"]),
            "ltp": float(tick["close"]),
            "source": "historical_replay",
            "trace_id": f"replay:{tick['timestamp'].isoformat()}:{symbol}",
        }
        self._publish_quote(symbol, payload)
        self._dispatch_tick(symbol, payload)

    def _publish_quote(self, symbol: str, tick: Mapping[str, Any]) -> None:
        hub = getattr(self._paper, "hub", None)
        store_quote = getattr(hub, "store_quote", None)
        if callable(store_quote):
            store_quote(symbol, dict(tick), source="historical_replay")
            return
        quotes = getattr(hub, "quotes", None)
        if isinstance(quotes, dict):
            quotes[symbol] = dict(tick)

    def _dispatch_tick(self, symbol: str, tick: Mapping[str, Any]) -> None:
        for attr in ("on_tick_event", "on_datahub_tick", "_on_tick_safe"):
            handler = getattr(self._runner, attr, None)
            if callable(handler):
                handler(dict(tick))
                return
        handler = getattr(self._runner, "_on_tick", None)
        if callable(handler):
            handler(symbol, tick)
            return
        raise AttributeError(
            "StrategyRunner does not expose the production tick ingestion surface"
        )


def _build_tick(row: pd.Series, timestamp: datetime, *, prefix: str) -> dict[str, Any]:
    columns = {col for col in row.index if col.startswith(prefix)}
    if not columns:
        return {}

    def _pick(name: str, default: float = 0.0) -> float:
        return float(row.get(f"{prefix}{name}", default))

    tick = {
        "timestamp": timestamp,
        "open": _pick("open"),
        "high": _pick("high"),
        "low": _pick("low"),
        "close": _pick("close"),
        "volume": _pick("volume"),
        "oi": _pick("oi"),
    }
    for name in ("bid", "ask", "instrument_token"):
        value = row.get(f"{prefix}{name}")
        if value is not None and not pd.isna(value):
            tick[name] = float(value) if name != "instrument_token" else int(value)
    return tick


def _load_frame(path: Path) -> pd.DataFrame:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        msg = f"Replay source {resolved} not found"
        raise FileNotFoundError(msg)
    if resolved.suffix.lower() == ".csv":
        frame = pd.read_csv(resolved, index_col=0)
    else:
        try:
            frame = pd.read_parquet(resolved)
        except Exception as exc:  # noqa: BLE001 - surface import errors cleanly
            msg = "Parquet replay requires pandas parquet dependencies"
            raise RuntimeError(msg) from exc
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame.sort_index(inplace=True)
    return frame


def _resolve_day_path(directory: Path, day: str) -> Path:
    base = directory.expanduser().resolve()
    if not base.exists():
        msg = f"Replay directory {base} not found"
        raise FileNotFoundError(msg)
    candidates = [
        base / f"{day}.csv",
        base / f"{day}.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    msg = f"Replay day {day} missing under {base}"
    raise FileNotFoundError(msg)


def _collect_trade_history(runner: Any) -> list[dict[str, Any]]:
    if not hasattr(runner, "get_status"):
        return []
    try:
        status = runner.get_status()
    except Exception:
        return []
    symbols = status.get("symbols", {}) if isinstance(status, Mapping) else {}
    trades: list[dict[str, Any]] = []
    for symbol_state in symbols.values():
        history = (
            symbol_state.get("trade_history")
            if isinstance(symbol_state, Mapping)
            else None
        )
        if not history:
            continue
        for record in history:
            if hasattr(record, "to_dict"):
                trades.append(record.to_dict())
            elif isinstance(record, Mapping):
                trades.append(dict(record))
    return trades


__all__ = ["ReplayClock", "ReplayHarness", "ReplayResult"]
