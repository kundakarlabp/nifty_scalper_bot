"""Minute-bar replay harness for exercising the live trading stack."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from nifty_scalper_bot.execution.paper_fill_engine import PaperFillEngine

_IST = ZoneInfo("Asia/Kolkata")


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


@dataclass(frozen=True, slots=True)
class ReplayContractSnapshot:
    """Historical active basket recorded from ``available_from`` onward."""

    available_from: datetime
    basket: Mapping[str, Any]


class HistoricalContractCatalog:
    """Resolve the latest time-available basket without future-data look-ahead."""

    def __init__(self, snapshots: Sequence[ReplayContractSnapshot]) -> None:
        if not snapshots:
            raise ValueError("historical contract catalog requires snapshots")
        ordered = sorted(snapshots, key=lambda item: _as_utc(item.available_from))
        timestamps = [_as_utc(item.available_from) for item in ordered]
        if len(set(timestamps)) != len(timestamps):
            raise ValueError("historical contract snapshots require unique timestamps")
        self._snapshots = tuple(ordered)
        self._timestamps = tuple(timestamps)

    def resolve(self, timestamp: datetime) -> Mapping[str, Any]:
        """Return the basket known at ``timestamp`` or fail closed."""

        target = _as_utc(timestamp)
        index = bisect_right(self._timestamps, target) - 1
        if index < 0:
            raise LookupError(
                f"no contract basket available at {timestamp.isoformat()}"
            )
        basket = self._snapshots[index].basket
        expiry = _parse_contract_expiry(basket.get("option_expiry"))
        if expiry is None:
            raise ValueError("historical contract basket requires option_expiry")
        trading_day = target.astimezone(_IST).date()
        if expiry < trading_day:
            raise LookupError(
                f"historical contract basket expired on {expiry.isoformat()}"
            )
        return basket


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
        contract_catalog: HistoricalContractCatalog | None = None,
    ) -> None:
        self._runner = runner
        self._paper = paper_engine
        self._option_symbol = option_symbol
        self._index_symbol = index_symbol
        self._clock = clock
        self._contract_catalog = contract_catalog
        self._active_basket_key: tuple[Any, ...] | None = None
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
            option_symbol = self._option_symbol
            if self._contract_catalog is not None:
                basket = self._contract_catalog.resolve(tick_time)
                self._install_contract_basket(basket)
                option_symbol = _row_option_symbol(row, option_symbol)
                _require_available_option(option_symbol, basket, tick_time)
            if self._index_symbol:
                index_tick = _build_tick(row, tick_time, prefix="index_")
                if index_tick:
                    self._publish_and_dispatch(self._index_symbol, index_tick)
            option_tick = _build_tick(row, tick_time, prefix="option_")
            if option_tick:
                self._publish_and_dispatch(option_symbol, option_tick)
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

    def _install_contract_basket(self, basket: Mapping[str, Any]) -> None:
        key = (
            basket.get("basket_version"),
            basket.get("selected_ce"),
            basket.get("selected_ce_token"),
            basket.get("selected_pe"),
            basket.get("selected_pe_token"),
            basket.get("option_expiry"),
            tuple(basket.get("option_symbols") or ()),
            tuple(basket.get("option_tokens") or ()),
        )
        if key == self._active_basket_key:
            return
        set_runner_basket = getattr(self._runner, "set_active_trading_universe", None)
        if not callable(set_runner_basket):
            raise AttributeError(
                "StrategyRunner cannot install historical active contract basket"
            )
        set_runner_basket(basket)
        hub = getattr(self._paper, "hub", None)
        set_hub_basket = getattr(hub, "set_active_contract_basket", None)
        if not callable(set_hub_basket):
            raise AttributeError(
                "PaperFillEngine quote hub cannot install historical contract basket"
            )
        set_hub_basket(basket)
        self._active_basket_key = key


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


def _as_utc(value: datetime) -> datetime:
    timestamp = value if value.tzinfo is not None else value.replace(tzinfo=_IST)
    return timestamp.astimezone(ZoneInfo("UTC"))


def _parse_contract_expiry(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _row_option_symbol(row: pd.Series, default: str) -> str:
    value = row.get("option_symbol")
    if value is None or pd.isna(value):
        return default
    symbol = str(value).strip()
    return symbol or default


def _require_available_option(
    symbol: str,
    basket: Mapping[str, Any],
    timestamp: datetime,
) -> None:
    available = {
        str(item).strip().upper()
        for item in (
            basket.get("option_symbols")
            or (basket.get("selected_ce"), basket.get("selected_pe"))
        )
        if item
    }
    normalized = str(symbol).strip().upper()
    if normalized not in available:
        raise ValueError(
            f"option {symbol} not present in historical basket at "
            f"{timestamp.isoformat()}"
        )


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


__all__ = [
    "HistoricalContractCatalog",
    "ReplayClock",
    "ReplayContractSnapshot",
    "ReplayHarness",
    "ReplayResult",
]
