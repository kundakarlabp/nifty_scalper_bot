"""Comprehensive vector/backtest execution utilities."""

from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import pandas as pd

from plotly import graph_objects as go
from plotly.offline import plot

try:  # pragma: no cover - optional zoneinfo import for <3.9
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - defensive fallback
    ZoneInfo = None  # type: ignore[misc, assignment]

if ZoneInfo is None:
    _INDIA_TZ: object | None = None
else:
    _INDIA_TZ = ZoneInfo("Asia/Kolkata")

from nifty_scalper_bot.strategies.signal_generator import (
    Position as StrategyPosition,
)
from nifty_scalper_bot.utils.logging import get_logger

if TYPE_CHECKING:
    from nifty_scalper_bot.backtest.parity import SinglePipelineParity


LOGGER = get_logger(__name__)


class StrategyProtocol(Protocol):
    """Protocol describing the minimum surface of a strategy used for backtests."""

    name: str

    def generate_signals(
        self, market_data: pd.DataFrame
    ) -> pd.Series:  # pragma: no cover - interface
        """Return a Series of target signals indexed by timestamp."""


@dataclass(slots=True)
class BacktestConfig:
    """Configuration values controlling the simulation."""

    initial_cash: float = 100_000.0
    slippage_pct: float = 0.0005
    commission_pct: float = 0.0005
    risk_free_rate: float = 0.02
    price_column: str = "close"
    allow_short: bool = True
    position_mode: str = "fixed"  # "fixed" or "percent"
    fixed_quantity: int = 1
    percent_of_equity: float = 0.1
    output_directory: Path = Path("backtest_reports")
    generate_visualizations: bool = True

    def validate(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 <= self.slippage_pct < 0.05:
            raise ValueError("slippage_pct must be between 0 and 0.05")
        if not 0 <= self.commission_pct < 0.05:
            raise ValueError("commission_pct must be between 0 and 0.05")
        if self.position_mode not in {"fixed", "percent"}:
            raise ValueError("position_mode must be 'fixed' or 'percent'")
        if self.position_mode == "fixed" and self.fixed_quantity <= 0:
            raise ValueError("fixed_quantity must be positive")
        if self.position_mode == "percent" and not 0 < self.percent_of_equity <= 1:
            raise ValueError("percent_of_equity must be within (0, 1]")


@dataclass(slots=True)
class Order:
    """Individual order executed during the backtest."""

    timestamp: pd.Timestamp
    side: Literal["BUY", "SELL"]
    price: float
    quantity: int
    commission: float
    cash_balance: float
    target_price: float | None = None
    slippage: float | None = None
    latency: float | None = None


@dataclass(slots=True)
class Trade:
    """Aggregated trade information from entry to exit."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    duration: timedelta
    entry_commission: float
    exit_commission: float


@dataclass
class BacktestResult:
    """Holds the outcome of a backtest run."""

    equity_curve: pd.DataFrame
    performance: Dict[str, float]
    trades: List[Trade]
    orders: List[Order]
    config: BacktestConfig
    report_path: Optional[Path] = None
    visualization_path: Optional[Path] = None
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    analytics: Dict[str, object] | None = None
    signals: pd.Series | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "performance": self.performance,
            "trades": [trade.__dict__ for trade in self.trades],
            "orders": [order.__dict__ for order in self.orders],
            "analytics": self.analytics or {},
            "json_report": str(self.json_path) if self.json_path else None,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        """Return a serialisable snapshot of trades, orders, and signals.

        Args:
            None.

        Returns:
            dict[str, Any]: Snapshot payload suitable for regression baselines.

        Raises:
            RuntimeError: If serialisation fails unexpectedly.
        """

        LOGGER.debug(
            "Entered BacktestResult.to_snapshot",
            extra={"event": "backtest_result_snapshot_enter"},
        )
        try:
            trades_payload = [
                {
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "direction": trade.direction,
                    "quantity": int(trade.quantity),
                    "entry_price": float(trade.entry_price),
                    "exit_price": float(trade.exit_price),
                    "pnl": float(trade.pnl),
                    "return_pct": float(trade.return_pct),
                }
                for trade in self.trades
            ]
            orders_payload = [
                {
                    "timestamp": order.timestamp.isoformat(),
                    "side": order.side,
                    "price": float(order.price),
                    "quantity": int(order.quantity),
                    "commission": float(order.commission),
                }
                for order in self.orders
            ]
            signal_payload: Dict[str, int] = {}
            if isinstance(self.signals, pd.Series):
                for timestamp, value in self.signals.items():
                    key = pd.Timestamp(timestamp).isoformat()
                    signal_payload[key] = int(value)
            snapshot: Dict[str, Any] = {
                "performance": {
                    key: float(value) for key, value in self.performance.items()
                },
                "trades": trades_payload,
                "orders": orders_payload,
                "signals": signal_payload,
            }
            return snapshot
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult.to_snapshot: %s",
                exc,
                extra={"event": "backtest_result_snapshot_error"},
            )
            raise

    def diff_against(self, baseline: Mapping[str, Any] | None) -> Dict[str, Any]:
        """Return drift summary between the current result and ``baseline``.

        Args:
            baseline: Previously recorded snapshot used for regression checks.

        Returns:
            dict[str, Any]: Drift summary including mismatched signals and trades.

        Raises:
            RuntimeError: If comparison cannot be completed.
        """

        LOGGER.debug(
            "Entered BacktestResult.diff_against",
            extra={"event": "backtest_result_diff_enter"},
        )
        try:
            if baseline is None:
                return {
                    "has_drift": False,
                    "signals": [],
                    "trades": [],
                    "orders": [],
                    "performance_delta": {},
                }
            snapshot = self.to_snapshot()
            baseline_trades = list(baseline.get("trades", []))
            baseline_orders = list(baseline.get("orders", []))
            baseline_signals = (
                baseline.get("signals", {})
                if isinstance(baseline.get("signals", {}), Mapping)
                else {}
            )
            baseline_performance = (
                baseline.get("performance", {})
                if isinstance(baseline.get("performance", {}), Mapping)
                else {}
            )

            trade_drift: List[Dict[str, Any]] = []
            current_trades = list(snapshot.get("trades", []))
            for index, expected in enumerate(baseline_trades):
                if index >= len(current_trades):
                    trade_drift.append(
                        {"index": index, "expected": expected, "actual": None}
                    )
                    continue
                actual = current_trades[index]
                if actual != expected:
                    trade_drift.append(
                        {"index": index, "expected": expected, "actual": actual}
                    )
            if len(current_trades) > len(baseline_trades):
                for index in range(len(baseline_trades), len(current_trades)):
                    trade_drift.append(
                        {
                            "index": index,
                            "expected": None,
                            "actual": current_trades[index],
                        }
                    )

            order_drift: List[Dict[str, Any]] = []
            current_orders = list(snapshot.get("orders", []))
            for index, expected in enumerate(baseline_orders):
                if index >= len(current_orders):
                    order_drift.append(
                        {"index": index, "expected": expected, "actual": None}
                    )
                    continue
                actual = current_orders[index]
                if actual != expected:
                    order_drift.append(
                        {"index": index, "expected": expected, "actual": actual}
                    )
            if len(current_orders) > len(baseline_orders):
                for index in range(len(baseline_orders), len(current_orders)):
                    order_drift.append(
                        {
                            "index": index,
                            "expected": None,
                            "actual": current_orders[index],
                        }
                    )

            signal_drift: List[Dict[str, Any]] = []
            current_signals = (
                snapshot.get("signals", {})
                if isinstance(snapshot.get("signals", {}), Mapping)
                else {}
            )
            for timestamp in sorted(set(current_signals) | set(baseline_signals)):
                actual = current_signals.get(timestamp)
                expected = baseline_signals.get(timestamp)
                if actual != expected:
                    signal_drift.append(
                        {
                            "timestamp": timestamp,
                            "expected": expected,
                            "actual": actual,
                        }
                    )

            performance_delta: Dict[str, float] = {}
            for key in set(baseline_performance) | set(self.performance):
                current_value = float(self.performance.get(key, 0.0))
                baseline_value = float(baseline_performance.get(key, 0.0))
                delta = current_value - baseline_value
                if abs(delta) > 1e-9:
                    performance_delta[key] = delta

            has_drift = bool(
                signal_drift or trade_drift or order_drift or performance_delta
            )
            if has_drift:
                LOGGER.error(
                    "Condition met: backtest_result_drift_detected",
                    extra={
                        "event": "backtest_result_drift_detected",
                        "signal_drift": len(signal_drift),
                        "trade_drift": len(trade_drift),
                        "order_drift": len(order_drift),
                    },
                )
            else:
                LOGGER.info(
                    "Condition met: backtest_result_baseline_match",
                    extra={"event": "backtest_result_baseline_match"},
                )
            return {
                "has_drift": has_drift,
                "signals": signal_drift,
                "trades": trade_drift,
                "orders": order_drift,
                "performance_delta": performance_delta,
            }
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult.diff_against: %s",
                exc,
                extra={"event": "backtest_result_diff_error"},
            )
            raise

    def metrics_summary(self) -> Dict[str, Any]:
        """Return aggregated metrics for downstream reporting.

        Args:
            None.

        Returns:
            dict[str, Any]: Daily PnL, drawdown, and latency statistics.

        Raises:
            RuntimeError: If metrics aggregation fails.
        """

        LOGGER.debug(
            "Entered BacktestResult.metrics_summary",
            extra={"event": "backtest_result_metrics_enter"},
        )
        try:
            equity_series = self.equity_curve.get("equity", pd.Series(dtype=float))
            daily_pnl: Dict[str, float] = {}
            if not equity_series.empty:
                daily_window = (
                    equity_series.resample("1D").agg(["first", "last"]).dropna()
                )
                for timestamp, row in daily_window.iterrows():
                    daily_pnl[str(pd.Timestamp(timestamp).date())] = float(
                        row["last"] - row["first"]
                    )
            latency_stats = self.analytics or {}
            error_events = 0
            if isinstance(latency_stats, Mapping):
                parity_summary = latency_stats.get("pipeline_parity")
                if isinstance(parity_summary, Mapping):
                    diff_blob = parity_summary.get("diff")
                    if isinstance(diff_blob, str) and diff_blob.strip():
                        error_events += 1
                    live_orders = parity_summary.get("live_orders")
                    back_orders = parity_summary.get("backtest_orders")
                    error_events += self._count_sequence_mismatches(
                        live_orders, back_orders
                    )
                    live_json = parity_summary.get("live_json")
                    back_json = parity_summary.get("back_json")
                    error_events += self._count_trade_mismatches(live_json, back_json)
                raw_errors = latency_stats.get("errors")
                error_events += self._count_generic_errors(raw_errors)
            summary = {
                "daily_pnl": daily_pnl,
                "total_pnl": (
                    float(equity_series.iloc[-1] - equity_series.iloc[0])
                    if not equity_series.empty
                    else 0.0
                ),
                "max_drawdown": float(self.performance.get("max_drawdown", 0.0)),
                "order_latency": (
                    latency_stats.get("order_latency", {})
                    if isinstance(latency_stats, Mapping)
                    else {}
                ),
                "tick_latency": (
                    latency_stats.get("tick_latency", {})
                    if isinstance(latency_stats, Mapping)
                    else {}
                ),
                "error_events": int(error_events),
            }
            return summary
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult.metrics_summary: %s",
                exc,
                extra={"event": "backtest_result_metrics_error"},
            )
            raise

    def _count_sequence_mismatches(
        self,
        first: object,
        second: object,
    ) -> int:
        """Return count of mismatched entries between two sequences.

        Args:
            first: Sequence representing the observed data.
            second: Sequence representing the baseline data.

        Returns:
            int: Number of mismatched or missing elements across both inputs.

        Raises:
            RuntimeError: If sequence comparison fails unexpectedly.
        """

        LOGGER.debug(
            "Entered BacktestResult._count_sequence_mismatches",
            extra={"event": "backtest_result_count_seq_enter"},
        )
        try:
            seq_first = (
                list(first)
                if isinstance(first, Sequence) and not isinstance(first, (str, bytes))
                else []
            )
            seq_second = (
                list(second)
                if isinstance(second, Sequence) and not isinstance(second, (str, bytes))
                else []
            )
            limit = max(len(seq_first), len(seq_second))
            mismatches = 0
            for index in range(limit):
                left = seq_first[index] if index < len(seq_first) else None
                right = seq_second[index] if index < len(seq_second) else None
                if left != right:
                    mismatches += 1
            return mismatches
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult._count_sequence_mismatches: %s",
                exc,
                extra={"event": "backtest_result_count_seq_error"},
            )
            raise

    def _count_trade_mismatches(
        self,
        live_payload: object,
        back_payload: object,
    ) -> int:
        """Return mismatched trade entries decoded from JSON payloads.

        Args:
            live_payload: Serialized or in-memory trade payload from live replay.
            back_payload: Serialized or in-memory trade payload from backtest replay.

        Returns:
            int: Number of mismatched trades across both payloads.

        Raises:
            RuntimeError: If payload decoding fails unexpectedly.
        """

        LOGGER.debug(
            "Entered BacktestResult._count_trade_mismatches",
            extra={"event": "backtest_result_count_trade_enter"},
        )
        try:
            live_trades: List[Any]
            if isinstance(live_payload, str):
                decoded = json.loads(live_payload)
                live_trades = decoded if isinstance(decoded, list) else []
            elif isinstance(live_payload, Sequence) and not isinstance(
                live_payload, (str, bytes)
            ):
                live_trades = list(live_payload)
            else:
                live_trades = []

            back_trades: List[Any]
            if isinstance(back_payload, str):
                decoded_back = json.loads(back_payload)
                back_trades = decoded_back if isinstance(decoded_back, list) else []
            elif isinstance(back_payload, Sequence) and not isinstance(
                back_payload, (str, bytes)
            ):
                back_trades = list(back_payload)
            else:
                back_trades = []

            return self._count_sequence_mismatches(live_trades, back_trades)
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult._count_trade_mismatches: %s",
                exc,
                extra={"event": "backtest_result_count_trade_error"},
            )
            raise

    def _count_generic_errors(self, payload: object) -> int:
        """Return error count derived from analytics payloads.

        Args:
            payload: Analytics field containing error metadata.

        Returns:
            int: Total number of reported errors.

        Raises:
            RuntimeError: If coercion or counting fails unexpectedly.
        """

        LOGGER.debug(
            "Entered BacktestResult._count_generic_errors",
            extra={"event": "backtest_result_count_generic_enter"},
        )
        try:
            if isinstance(payload, Mapping):
                total = 0
                for value in payload.values():
                    if isinstance(value, (int, float)):
                        total += int(abs(value))
                return total
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                return len(list(payload))
            if isinstance(payload, (int, float)):
                return int(abs(payload))
            return 0
        except Exception as exc:  # noqa: BLE001 - defensive wrapping
            LOGGER.error(
                "Failure in BacktestResult._count_generic_errors: %s",
                exc,
                extra={"event": "backtest_result_count_generic_error"},
            )
            raise


class BacktestEngine:
    """Run realistic backtests with order-by-order simulation and reporting."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: StrategyProtocol,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        if data.empty:
            raise ValueError("Market data is empty")
        if strategy is None:
            raise ValueError("strategy must be provided")
        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("data must be indexed by datetime")
        self.data = self.data.sort_index()
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.config.validate()
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self._current_trade: Optional[Trade] = None
        self._tick_latencies: list[float] = []
        self._order_latencies: list[float] = []
        self._last_order_timestamp: Optional[pd.Timestamp] = None

    def run(
        self,
        *,
        parity: "SinglePipelineParity" | None = None,
        parity_frame: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Execute the configured backtest and return the aggregated result.

        Args:
            parity: Optional pipeline parity harness mirroring live execution.
            parity_frame: Optional pre-built dataframe forwarded to the parity harness.

        Returns:
            BacktestResult: Simulation outcome including analytics and artefacts.

        Raises:
            Exception: Propagates errors from strategies, parity harness, or reporting.
        """

        LOGGER.debug(
            "Entered BacktestEngine.run",
            extra={"event": "backtest_engine_run_enter"},
        )
        try:
            self.orders.clear()
            self.trades.clear()
            self._current_trade = None
            self._tick_latencies.clear()
            self._order_latencies.clear()
            self._last_order_timestamp = None
            signals = self._prepare_signals(self.strategy.generate_signals(self.data))
            equity_curve = self._simulate_orders(signals)
            performance = self._calculate_performance(equity_curve)
            analytics = self._build_analytics(equity_curve)
            analytics["max_drawdown"] = performance.get("max_drawdown", 0.0)
            if parity is not None:
                parity_summary = self._run_pipeline_parity(parity, parity_frame)
                analytics["pipeline_parity"] = parity_summary
            report_dir = self.config.output_directory / "reporting"
            report_dir.mkdir(parents=True, exist_ok=True)
            visualization_path: Optional[Path] = None
            if self.config.generate_visualizations:
                visualization_path = self._create_equity_chart(equity_curve)
            report_date = pd.Timestamp.utcnow().strftime("%Y%m%d")
            report_path = self._build_html_report(
                equity_curve,
                performance,
                visualization_path,
                analytics,
                report_dir,
                report_date,
            )
            markdown_path = self._build_markdown_report(
                equity_curve, performance, analytics, report_dir, report_date
            )
            json_path = self._build_json_report(
                equity_curve, performance, analytics, report_dir, report_date
            )
        except Exception as exc:  # pragma: no cover - defensive logging surface
            LOGGER.error(
                "Failure in BacktestEngine.run: %s",
                exc,
                extra={"event": "backtest_engine_run_error"},
            )
            raise
        LOGGER.info(
            "Condition met: backtest_engine_run_complete",
            extra={"event": "backtest_engine_run_complete"},
        )
        return BacktestResult(
            equity_curve=equity_curve,
            performance=performance,
            trades=self.trades,
            orders=self.orders,
            config=self.config,
            report_path=report_path,
            visualization_path=visualization_path,
            markdown_path=markdown_path,
            json_path=json_path,
            analytics=analytics,
            signals=signals,
        )

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _prepare_signals(self, signals: pd.Series) -> pd.Series:
        if not isinstance(signals, pd.Series):
            raise TypeError("Strategy generate_signals must return a pandas Series")
        if not signals.index.equals(self.data.index):
            signals = signals.reindex(self.data.index).fillna(0)
        allowed_values = {-1, 0, 1}
        invalid = set(signals.dropna().unique()) - allowed_values
        if invalid:
            raise ValueError(f"Signals contain unsupported values: {invalid}")
        return signals.astype(int)

    def _simulate_orders(self, signals: pd.Series) -> pd.DataFrame:
        cash = self.config.initial_cash
        position = 0
        equity_history: List[Tuple[pd.Timestamp, float, float, int]] = []
        previous_timestamp: pd.Timestamp | None = None
        for timestamp, row in self.data.iterrows():
            if previous_timestamp is not None:
                latency = (timestamp - previous_timestamp).total_seconds()
                if latency >= 0:
                    self._tick_latencies.append(latency)
            previous_timestamp = timestamp
            signal = signals.loc[timestamp]
            price = float(row[self.config.price_column])
            target_position = self._target_position(signal, cash, position, price)

            if (
                position != 0
                and target_position != 0
                and math.copysign(1, position) != math.copysign(1, target_position)
            ):
                # Flip: close existing first
                self._execute_order(timestamp, price, -position, cash, position)
                cash = self.orders[-1].cash_balance
                position = 0

            if target_position != position:
                self._execute_order(
                    timestamp, price, target_position - position, cash, position
                )
                cash = self.orders[-1].cash_balance
                position = target_position

            equity = cash + position * price
            equity_history.append((timestamp, equity, cash, position))

        equity_df = pd.DataFrame(
            equity_history, columns=["timestamp", "equity", "cash", "position"]
        ).set_index("timestamp")
        equity_df["returns"] = equity_df["equity"].pct_change().fillna(0.0)
        return equity_df

    def _target_position(
        self,
        signal: int,
        cash: float,
        current_position: int,
        price: float,
    ) -> int:
        if signal == 0:
            return 0
        if signal == -1 and not self.config.allow_short:
            return 0
        equity = cash + current_position * price
        quantity = self.config.fixed_quantity
        if self.config.position_mode == "percent":
            allocation = equity * self.config.percent_of_equity
            quantity = max(int(allocation // price), 0)
        return int(quantity * signal)

    def _execute_order(
        self,
        timestamp: pd.Timestamp,
        price: float,
        quantity_change: int,
        cash: float,
        current_position: int,
    ) -> None:
        if quantity_change == 0:
            return

        side: Literal["BUY", "SELL"] = "BUY" if quantity_change > 0 else "SELL"
        latency_value: float | None = None
        if self._last_order_timestamp is not None:
            latency_value = max(
                (timestamp - self._last_order_timestamp).total_seconds(), 0.0
            )
        self._last_order_timestamp = timestamp
        slip_multiplier = (
            1 + self.config.slippage_pct
            if side == "BUY"
            else 1 - self.config.slippage_pct
        )
        exec_price = price * slip_multiplier
        quantity = abs(quantity_change)
        trade_value = exec_price * quantity
        commission = trade_value * self.config.commission_pct
        slippage = exec_price - price

        if side == "BUY":
            cash -= trade_value + commission
        else:
            cash += trade_value - commission

        new_position = current_position + quantity_change
        self.orders.append(
            Order(
                timestamp=timestamp,
                side=side,
                price=exec_price,
                quantity=quantity,
                commission=commission,
                cash_balance=cash,
                target_price=price,
                slippage=slippage,
                latency=latency_value,
            )
        )
        if latency_value is not None:
            self._order_latencies.append(latency_value)

        self._update_trades(
            timestamp=timestamp,
            exec_price=exec_price,
            quantity_change=quantity_change,
            commission=commission,
            new_position=new_position,
            previous_position=current_position,
        )

    def _update_trades(
        self,
        timestamp: pd.Timestamp,
        exec_price: float,
        quantity_change: int,
        commission: float,
        new_position: int,
        previous_position: int,
    ) -> None:
        if previous_position == 0 and new_position != 0:
            direction = "LONG" if new_position > 0 else "SHORT"
            self._current_trade = Trade(
                entry_time=timestamp,
                exit_time=timestamp,
                direction=direction,
                quantity=abs(new_position),
                entry_price=exec_price,
                exit_price=exec_price,
                pnl=0.0,
                return_pct=0.0,
                duration=timedelta(0),
                entry_commission=commission,
                exit_commission=0.0,
            )
        elif new_position == 0 and previous_position != 0 and self._current_trade:
            exit_direction = "LONG" if previous_position > 0 else "SHORT"
            pnl, return_pct = self._calculate_trade_pnl(
                self._current_trade.entry_price,
                exec_price,
                abs(previous_position),
                exit_direction,
                self._current_trade.entry_commission,
                commission,
            )
            self.trades.append(
                Trade(
                    entry_time=self._current_trade.entry_time,
                    exit_time=timestamp,
                    direction=self._current_trade.direction,
                    quantity=self._current_trade.quantity,
                    entry_price=self._current_trade.entry_price,
                    exit_price=exec_price,
                    pnl=pnl,
                    return_pct=return_pct,
                    duration=timestamp - self._current_trade.entry_time,
                    entry_commission=self._current_trade.entry_commission,
                    exit_commission=commission,
                )
            )
            self._current_trade = None

    def _calculate_trade_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
        direction: str,
        entry_commission: float,
        exit_commission: float,
    ) -> Tuple[float, float]:
        gross = (exit_price - entry_price) * quantity
        if direction == "SHORT":
            gross = (entry_price - exit_price) * quantity
        net = gross - entry_commission - exit_commission
        return_pct = net / (entry_price * quantity) if entry_price else 0.0
        return net, return_pct

    def _calculate_performance(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        returns = equity_curve["returns"]
        total_return = (
            equity_curve["equity"].iloc[-1] / equity_curve["equity"].iloc[0] - 1
        )
        periods = len(equity_curve)
        annual_factor = 252
        cagr = (1 + total_return) ** (annual_factor / max(periods, 1)) - 1
        excess_returns = returns - self.config.risk_free_rate / annual_factor
        volatility = returns.std() * math.sqrt(annual_factor)
        sharpe = (
            (excess_returns.mean() * annual_factor) / volatility if volatility else 0.0
        )
        drawdown = self._calculate_drawdown(equity_curve["equity"])
        sortino = 0.0
        downside = returns[returns < 0].std() * math.sqrt(annual_factor)
        if downside:
            sortino = (returns.mean() * annual_factor) / downside
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        avg_trade_duration = self._calculate_average_trade_duration()
        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "volatility": volatility,
            "max_drawdown": drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_duration_days": avg_trade_duration,
        }

    def _calculate_drawdown(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        return float(drawdowns.min()) if not drawdowns.empty else 0.0

    def _calculate_win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for trade in self.trades if trade.pnl > 0)
        return wins / len(self.trades)

    def _calculate_profit_factor(self) -> float:
        gains = sum(trade.pnl for trade in self.trades if trade.pnl > 0)
        losses = sum(trade.pnl for trade in self.trades if trade.pnl < 0)
        if not losses:
            return float("inf") if gains > 0 else 0.0
        return gains / abs(losses)

    def _calculate_average_trade_duration(self) -> float:
        if not self.trades:
            return 0.0
        durations = [trade.duration for trade in self.trades]
        return (
            statistics.mean(duration.total_seconds() for duration in durations) / 86_400
        )

    def _build_analytics(self, equity_curve: pd.DataFrame) -> Dict[str, object]:
        tick_series = (
            pd.Series(self._tick_latencies)
            if self._tick_latencies
            else pd.Series(dtype=float)
        )
        order_series = (
            pd.Series(self._order_latencies)
            if self._order_latencies
            else pd.Series(dtype=float)
        )
        slippage_bps: list[float] = []
        for order in self.orders:
            if order.slippage is None or not order.target_price:
                continue
            try:
                bps = (order.slippage / order.target_price) * 10_000.0
            except ZeroDivisionError:
                continue
            slippage_bps.append(bps)
        risk_stats = {
            "avg_position": float(equity_curve["position"].abs().mean()),
            "max_position": float(equity_curve["position"].abs().max()),
            "cash_min": float(equity_curve["cash"].min()),
            "cash_max": float(equity_curve["cash"].max()),
        }
        analytics: Dict[str, object] = {
            "tick_latency": {
                "p50": None if tick_series.empty else float(tick_series.quantile(0.5)),
                "p95": None if tick_series.empty else float(tick_series.quantile(0.95)),
                "max": None if tick_series.empty else float(tick_series.max()),
            },
            "order_latency": {
                "p50": (
                    None if order_series.empty else float(order_series.quantile(0.5))
                ),
                "p95": (
                    None if order_series.empty else float(order_series.quantile(0.95))
                ),
                "max": None if order_series.empty else float(order_series.max()),
            },
            "slippage_bps": {
                "avg": (
                    None
                    if not slippage_bps
                    else float(sum(slippage_bps) / len(slippage_bps))
                ),
                "p95": (
                    None
                    if not slippage_bps
                    else float(pd.Series(slippage_bps).quantile(0.95))
                ),
            },
            "risk": risk_stats,
        }
        return analytics

    def _run_pipeline_parity(
        self,
        parity: "SinglePipelineParity",
        parity_frame: pd.DataFrame | None,
    ) -> Dict[str, object]:
        """Execute the parity harness and return the resulting summary.

        Args:
            parity: Parity harness ensuring live/backtest pipeline alignment.
            parity_frame: Optional replay dataframe overriding auto-generated payload.

        Returns:
            dict: Summary containing parity diff and serialized trade/order logs.

        Raises:
            Exception: Propagates errors raised by the parity harness.
        """

        LOGGER.debug(
            "Entered BacktestEngine._run_pipeline_parity",
            extra={"event": "backtest_engine_parity_enter"},
        )
        try:
            frame = (
                parity_frame.copy()
                if parity_frame is not None
                else self._build_parity_frame()
            )
            result = parity.run_frame(frame)
            live_json = json.dumps(
                result.live.trades,
                sort_keys=True,
                default=str,
                ensure_ascii=False,
            )
            back_json = json.dumps(
                result.backtest.trades,
                sort_keys=True,
                default=str,
                ensure_ascii=False,
            )
            summary: Dict[str, object] = {
                "diff": result.diff,
                "live_json": live_json,
                "back_json": back_json,
                "live_orders": result.live.orders,
                "backtest_orders": result.backtest.orders,
            }
        except Exception as exc:  # pragma: no cover - defensive logging surface
            LOGGER.error(
                "Failure in BacktestEngine._run_pipeline_parity: %s",
                exc,
                extra={"event": "backtest_engine_parity_error"},
            )
            raise
        if result.diff:
            LOGGER.error(
                "Condition met: backtest_engine_parity_mismatch",
                extra={
                    "event": "backtest_engine_parity_mismatch",
                    "diff": result.diff,
                },
            )
        else:
            LOGGER.info(
                "Condition met: backtest_engine_parity_match",
                extra={"event": "backtest_engine_parity_match"},
            )
        return summary

    def _build_parity_frame(self, override: pd.DataFrame | None = None) -> pd.DataFrame:
        """Construct a replay dataframe compatible with the parity harness.

        Args:
            override: Optional dataframe supplied directly to the parity harness.

        Returns:
            pd.DataFrame: Replay dataframe with option-prefixed OHLCV columns.

        Raises:
            ValueError: If the resulting parity frame is empty.
            Exception: Propagates pandas conversion errors.
        """

        LOGGER.debug(
            "Entered BacktestEngine._build_parity_frame",
            extra={"event": "backtest_engine_parity_frame_enter"},
        )
        try:
            if override is not None:
                frame = override.copy()
            else:
                option_columns = {}
                for column in ("open", "high", "low", "close", "volume", "oi"):
                    if column in self.data.columns:
                        option_columns[f"option_{column}"] = self.data[column]
                frame = pd.DataFrame(option_columns, index=self.data.index)
            if frame.empty:
                raise ValueError("Parity frame is empty")
            if not isinstance(frame.index, pd.DatetimeIndex):
                frame.index = pd.to_datetime(frame.index)
            frame.sort_index(inplace=True)
            return frame
        except Exception as exc:  # pragma: no cover - defensive logging surface
            LOGGER.error(
                "Failure in BacktestEngine._build_parity_frame: %s",
                exc,
                extra={"event": "backtest_engine_parity_frame_error"},
            )
            raise

    def _build_json_report(
        self,
        equity_curve: pd.DataFrame,
        performance: Dict[str, float],
        analytics: Dict[str, object],
        report_dir: Path,
        report_date: str,
    ) -> Path:
        """Write a JSON summary capturing daily PnL, drawdown, and latency metrics.

        Args:
            equity_curve: Equity history produced by the simulation.
            performance: Aggregate performance metrics computed by the engine.
            analytics: Aggregate analytics including latency and risk statistics.
            report_dir: Directory receiving reporting artefacts.
            report_date: Date token used for the artefact file name.

        Returns:
            Path: Location of the generated JSON summary.

        Raises:
            Exception: Propagates filesystem or pandas errors.
        """

        LOGGER.debug(
            "Entered BacktestEngine._build_json_report",
            extra={"event": "backtest_engine_json_report_enter"},
        )
        try:
            equity_series = equity_curve.get("equity", pd.Series(dtype=float))
            daily_pnl: Dict[str, float] = {}
            if not equity_series.empty:
                daily_window = (
                    equity_series.resample("1D").agg(["first", "last"]).dropna()
                )
                for timestamp, row in daily_window.iterrows():
                    daily_pnl[str(timestamp.date())] = float(row["last"] - row["first"])
                total_pnl = float(equity_series.iloc[-1] - equity_series.iloc[0])
            else:
                total_pnl = 0.0
            summary_payload = {
                "pnl": {
                    "daily": daily_pnl,
                    "total": total_pnl,
                },
                "latency": analytics.get("order_latency", {}),
                "tick_latency": analytics.get("tick_latency", {}),
                "drawdown": {
                    "max": float(performance.get("max_drawdown", 0.0)),
                },
                "risk": analytics.get("risk", {}),
                "generated_at": pd.Timestamp.utcnow().isoformat(),
            }
            json_path = report_dir / f"PnL_{report_date}.json"
            json_path.write_text(
                json.dumps(summary_payload, default=float, indent=2),
                encoding="utf-8",
            )
            return json_path
        except Exception as exc:  # pragma: no cover - defensive logging surface
            LOGGER.error(
                "Failure in BacktestEngine._build_json_report: %s",
                exc,
                extra={"event": "backtest_engine_json_report_error"},
            )
            raise

    def _create_equity_chart(self, equity_curve: pd.DataFrame) -> Path:
        output_dir = self.config.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=equity_curve.index, y=equity_curve["equity"], name="Equity Curve"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.data.index, y=self.data[self.config.price_column], name="Price"
            )
        )
        fig.update_layout(
            title=f"Backtest - {getattr(self.strategy, 'name', 'Strategy')}",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            legend=dict(orientation="h"),
        )
        file_path = output_dir / "equity_curve.html"
        chart_html = plot(fig, include_plotlyjs="cdn", output_type="div")
        file_path.write_text(chart_html, encoding="utf-8")
        return file_path

    def _build_html_report(
        self,
        equity_curve: pd.DataFrame,
        performance: Dict[str, float],
        visualization_path: Path | None,
        analytics: Dict[str, object],
        report_dir: Path,
        report_date: str,
    ) -> Path:
        metrics_html = "".join(
            f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
            for key, value in performance.items()
        )
        trades_rows = "".join(
            "<tr>"
            f"<td>{trade.entry_time}</td>"
            f"<td>{trade.exit_time}</td>"
            f"<td>{trade.direction}</td>"
            f"<td>{trade.quantity}</td>"
            f"<td>{trade.entry_price:.2f}</td>"
            f"<td>{trade.exit_price:.2f}</td>"
            f"<td>{trade.pnl:.2f}</td>"
            f"<td>{trade.return_pct:.4f}</td>"
            "</tr>"
            for trade in self.trades
        )
        equity_stats = equity_curve.describe().to_html(classes="table table-striped")
        chart_html = "<p>Visualization disabled.</p>"
        if visualization_path is not None and visualization_path.exists():
            with visualization_path.open("r", encoding="utf-8") as handle:
                chart_html = handle.read()

        def _render_section(title: str, payload: Mapping[str, object]) -> str:
            rows = "".join(
                (
                    f"<tr><td>{name}</td>"
                    f"<td>{(value if value is not None else 'n/a')}</td></tr>"
                )
                for name, value in payload.items()
            )
            return f"<h3>{title}</h3><table>{rows}</table>"

        latency_sections = "".join(
            _render_section(label.replace("_", " ").title(), stats)
            for label, stats in analytics.items()
            if isinstance(stats, Mapping)
            and label in {"tick_latency", "order_latency", "slippage_bps", "risk"}
        )
        strategy_name = getattr(self.strategy, "name", "Strategy")
        report_html = f"""
        <html>
            <head>
                <title>Backtest Report - {strategy_name}</title>
                <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/water.css@2/out/water.css\">
            </head>
            <body>
                <h1>Backtest Report - {strategy_name}</h1>
                <section>
                    <h2>Summary Metrics</h2>
                    <table>{metrics_html}</table>
                </section>
                <section>
                    <h2>Latency & Risk Analytics</h2>
                    {latency_sections}
                </section>
                <section>
                    <h2>Equity & Cash Statistics</h2>
                    {equity_stats}
                </section>
                <section>
                    <h2>Interactive Chart</h2>
                    {chart_html}
                </section>
                <section>
                    <h2>Trades</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Entry</th><th>Exit</th><th>Direction</th><th>Quantity</th>
                                <th>Entry Price</th><th>Exit Price</th><th>PNL</th>
                                <th>Return</th>
                            </tr>
                        </thead>
                        <tbody>{trades_rows}</tbody>
                    </table>
                </section>
            </body>
        </html>
        """
        report_path = report_dir / f"PnL_{report_date}.html"
        report_path.write_text(report_html, encoding="utf-8")
        return report_path

    def _build_markdown_report(
        self,
        equity_curve: pd.DataFrame,
        performance: Dict[str, float],
        analytics: Dict[str, object],
        report_dir: Path,
        report_date: str,
    ) -> Path:
        strategy_name = getattr(self.strategy, "name", "Strategy")
        lines = [f"# Backtest Report - {strategy_name}", ""]
        lines.append("## Summary Metrics")
        for key, value in performance.items():
            lines.append(f"- **{key}**: {value:.4f}")
        lines.append("")

        def _write_section(title: str, stats: Mapping[str, object]) -> None:
            lines.append(f"## {title}")
            for name, value in stats.items():
                formatted = "n/a" if value is None else f"{value}"
                lines.append(f"- **{name}**: {formatted}")
            lines.append("")

        for label in ("tick_latency", "order_latency", "slippage_bps", "risk"):
            payload = analytics.get(label)
            if isinstance(payload, Mapping):
                title = label.replace("_", " ").title()
                _write_section(title, payload)

        position_series = equity_curve["position"]
        lines.append("## Equity Summary")
        lines.append(
            f"- **Final Equity**: {equity_curve['equity'].iloc[-1]:.2f}\n"
            f"- **Max Position**: {position_series.abs().max():.0f}\n"
            f"- **Average Position**: {position_series.abs().mean():.2f}"
        )
        lines.append("")

        markdown_path = report_dir / f"PnL_{report_date}.md"
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return markdown_path


@dataclass(slots=True)
class BacktestSummary:
    """Compact metrics extracted from a quick backtest run."""

    symbol: str
    days: int
    total_pnl: float
    trade_count: int
    avg_trade_pnl: float
    win_rate_pct: float
    max_drawdown_pct: float
    strategy_breakdown: Dict[str, Dict[str, float]]
    result: BacktestResult

    def format_markdown(self) -> str:
        """Return Telegram-ready Markdown summary.

        Args:
            None.

        Returns:
            str: Markdown-formatted summary string.

        Raises:
            RuntimeError: If formatting fails unexpectedly.
        """

        LOGGER.debug(
            "Entered BacktestSummary.format_markdown",
            extra={"event": "backtest_summary_format_enter"},
        )
        try:
            lines = [
                f"*Backtest* `{self.symbol}` — last {self.days} day(s)",
                "",
                f"• Total PnL: `{self.total_pnl:.2f}`",
                f"• Trades: `{self.trade_count}`",
                f"• Avg trade PnL: `{self.avg_trade_pnl:.2f}`",
                f"• Win rate: `{self.win_rate_pct:.1f}%`",
                f"• Max drawdown: `{self.max_drawdown_pct:.2f}%`",
            ]
            if self.strategy_breakdown:
                lines.append("")
                lines.append("*Strategy breakdown*")
                for name, stats in sorted(self.strategy_breakdown.items()):
                    pnl = stats.get("pnl", 0.0)
                    trades = stats.get("trades", 0.0)
                    win_rate = stats.get("win_rate", 0.0)
                    lines.append(
                        " ".join(
                            [
                                f"`{name}`",
                                "→",
                                f"pnl `{pnl:.2f}`,",
                                f"trades `{trades:.0f}`,",
                                f"win `{win_rate:.1f}%`",
                            ]
                        )
                    )
            LOGGER.info(
                "Condition met: backtest_summary_format_complete",
                extra={"event": "backtest_summary_format_complete"},
            )
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - defensive logging
            LOGGER.error(
                "Failure in BacktestSummary.format_markdown: %s",
                exc,
                extra={"event": "backtest_summary_format_error"},
            )
            raise RuntimeError("Unable to format backtest summary") from exc


@dataclass(slots=True)
class _BacktestPosition(StrategyPosition):
    """Minimal position representation used by the RSI bridge."""

    symbol: str
    side: Literal["LONG", "SHORT"]
    quantity: int
    entry_price: float


class _RSIMeanReversionBridge(StrategyProtocol):
    """Adapt :class:`RSIMeanReversionStrategy` for :class:`BacktestEngine`."""

    def __init__(self, symbol: str) -> None:
        from nifty_scalper_bot.strategies.indicators import IndicatorEngine
        from nifty_scalper_bot.strategies.signal_generator import (
            RSIMeanReversionStrategy,
        )

        self.name = "RSI Mean Reversion"
        self._symbol = symbol
        self._indicator_engine = IndicatorEngine()
        self._strategy = RSIMeanReversionStrategy()
        self._position: StrategyPosition | None = None

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return {-1,0,1} signal series derived from RSI strategy.

        Args:
            market_data: Dataframe indexed by timestamp with OHLCV columns.

        Returns:
            pd.Series: Signal series aligned with *market_data* index.

        Raises:
            RuntimeError: If signal computation fails.
        """

        LOGGER.debug(
            "Entered _RSIMeanReversionBridge.generate_signals",
            extra={"event": "rsi_bridge_generate_enter", "rows": len(market_data)},
        )
        try:
            signals: list[int] = []
            for timestamp, row in market_data.iterrows():
                price = float(row.get("close", 0.0))
                volume_raw = row.get("volume", 0)
                try:
                    volume = int(float(volume_raw))
                except Exception:  # noqa: BLE001 - lenient casting
                    volume = 0
                ts = pd.Timestamp(timestamp)
                if ts.tzinfo is None:
                    if _INDIA_TZ is not None:
                        ts = ts.tz_localize(_INDIA_TZ)
                    else:  # pragma: no cover - fallback when zoneinfo missing
                        ts = ts.tz_localize(timezone.utc)
                bar_payload = {
                    "open": float(row.get("open", price)),
                    "high": float(row.get("high", price)),
                    "low": float(row.get("low", price)),
                    "close": price,
                }
                self._indicator_engine.update_price(
                    self._symbol,
                    bar_payload,
                    volume=volume,
                    timestamp=ts.to_pydatetime(),
                )
                indicators = self._indicator_engine.get_indicators(self._symbol)
                signal = self._strategy.generate_signal(
                    self._symbol,
                    indicators,
                    price,
                    self._position,
                )
                if signal is None:
                    signals.append(0)
                    continue
                action = signal.action
                if action == "BUY":
                    self._position = _BacktestPosition(
                        symbol=self._symbol,
                        side=cast(Literal["LONG", "SHORT"], "LONG"),
                        quantity=max(int(signal.quantity or 1), 1),
                        entry_price=price,
                    )
                    signals.append(1)
                elif action == "SELL":
                    self._position = _BacktestPosition(
                        symbol=self._symbol,
                        side=cast(Literal["LONG", "SHORT"], "SHORT"),
                        quantity=max(int(signal.quantity or 1), 1),
                        entry_price=price,
                    )
                    signals.append(-1)
                elif action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                    self._position = None
                    signals.append(0)
                else:
                    signals.append(0)
            series = pd.Series(signals, index=market_data.index, dtype=int)
            LOGGER.info(
                "Condition met: rsi_bridge_generate_complete",
                extra={
                    "event": "rsi_bridge_generate_complete",
                    "signals": len(signals),
                },
            )
            return series
        except Exception as exc:  # noqa: BLE001 - defensive logging
            LOGGER.error(
                "Failure in _RSIMeanReversionBridge.generate_signals: %s",
                exc,
                extra={"event": "rsi_bridge_generate_error"},
            )
            raise RuntimeError("RSI backtest signal generation failed") from exc


def _resolve_backtest_history_root(
    override: Path | None = None,
) -> Path:
    """Return the resolved directory containing quick backtest data.

    Args:
        override: Optional explicit directory provided by the caller.

    Returns:
        Path: Concrete directory containing historical data files.

    Raises:
        FileNotFoundError: If no candidate directory exists.
        RuntimeError: If resolution fails unexpectedly.
    """

    LOGGER.debug(
        "Entered _resolve_backtest_history_root",
        extra={"event": "resolve_history_root_enter"},
    )
    try:
        candidates: list[Path] = []
        if override is not None:
            candidates.append(override)
        env_vars = (
            "BACKTEST_HISTORY_DIR",
            "NIFTY_SCALPER_BACKTEST_DATA_DIR",
            "NIFTY_SCALPER_BOT_DATA_DIR",
            "CAB_DATA_DIR",
        )
        for name in env_vars:
            raw = os.getenv(name)
            if raw:
                candidates.append(Path(raw))
        candidates.extend(
            [
                Path("data/cab"),
                Path("data/history"),
                Path("data/options"),
                Path("data"),
                Path("backtests/data"),
            ]
        )
        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except FileNotFoundError:
                resolved = candidate.expanduser()
            if resolved in seen:
                continue
            seen.add(resolved)
            if not resolved.exists():
                continue
            selected = resolved if resolved.is_dir() else resolved.parent
            LOGGER.info(
                "Condition met: resolve_history_root_selected",
                extra={
                    "event": "resolve_history_root_selected",
                    "path": str(selected),
                },
            )
            return selected
        raise FileNotFoundError("Backtest data directory not found")
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - defensive logging
        LOGGER.error(
            "Failure in _resolve_backtest_history_root: %s",
            exc,
            extra={"event": "resolve_history_root_error"},
        )
        raise RuntimeError("Failed to resolve backtest data directory") from exc


def _discover_history_files(symbol: str, base_dir: Path) -> list[Path]:
    """Return available history files for *symbol* inside *base_dir*.

    Args:
        symbol: Canonical instrument identifier.
        base_dir: Directory containing historical data files.

    Returns:
        list[Path]: Sorted list of candidate files.

    Raises:
        FileNotFoundError: If the directory does not contain symbol data.
    """

    LOGGER.debug(
        "Entered _discover_history_files",
        extra={"event": "discover_history_enter", "symbol": symbol},
    )
    try:
        resolved = base_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Data directory {resolved} not found")
        symbol_upper = symbol.upper()
        search_roots: list[Path] = [resolved]
        for name in (symbol_upper, "cab", "history", "options"):
            candidate_dir = resolved / name
            if candidate_dir.exists() and candidate_dir.is_dir():
                search_roots.append(candidate_dir)
        candidates: dict[str, Path] = {}
        for root in search_roots:
            direct_dir = root / symbol_upper
            if direct_dir.exists() and direct_dir.is_dir():
                for path in direct_dir.iterdir():
                    if path.suffix.lower() in {".csv", ".parquet"}:
                        candidates[str(path)] = path
            patterns = [
                f"{symbol_upper}*.csv",
                f"{symbol_upper}*.parquet",
                f"*{symbol_upper}*.csv",
                f"*{symbol_upper}*.parquet",
                "nifty_options_all*.csv",
                "nifty_options_all*.parquet",
                "options_all*.csv",
                "options_all*.parquet",
            ]
            for pattern in patterns:
                for path in root.glob(pattern):
                    if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
                        candidates[str(path)] = path
        candidate_list = sorted(candidates.values())
        if not candidate_list:
            raise FileNotFoundError(
                f"No historical files found for symbol {symbol_upper} in {resolved}"
            )
        LOGGER.info(
            "Condition met: discover_history_complete",
            extra={
                "event": "discover_history_complete",
                "count": len(candidate_list),
            },
        )
        return candidate_list
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - defensive logging
        LOGGER.error(
            "Failure in _discover_history_files: %s",
            exc,
            extra={"event": "discover_history_error"},
        )
        raise RuntimeError("Unable to enumerate history files") from exc


def _normalise_history_frame(
    frame: pd.DataFrame, *, symbol: str, source: Path
) -> pd.DataFrame:
    """Return a timestamp-indexed frame filtered for *symbol*.

    Args:
        frame: Raw DataFrame loaded from disk.
        symbol: Uppercase symbol token to filter by.
        source: File path used for logging context.

    Returns:
        pd.DataFrame: Sanitised DataFrame indexed by timestamp.

    Raises:
        RuntimeError: If the frame cannot be normalised.
    """

    LOGGER.debug(
        "Entered _normalise_history_frame",
        extra={
            "event": "normalise_history_frame_enter",
            "rows": len(frame.index),
            "source": str(source),
        },
    )
    try:
        working = frame.copy()
        column_lookup = {col.lower(): col for col in working.columns}
        symbol_column = None
        for candidate in ("symbol", "tradingsymbol", "instrument"):
            if candidate in column_lookup:
                symbol_column = column_lookup[candidate]
                break
        if symbol_column is not None:
            working[symbol_column] = working[symbol_column].astype(str).str.upper()
            working = working[working[symbol_column] == symbol]
            working = working.drop(columns=[symbol_column])
        if working.empty:
            return working
        timestamp_column = None
        for candidate in ("timestamp", "datetime", "date_time", "time_stamp"):
            if candidate in column_lookup:
                timestamp_column = column_lookup[candidate]
                break
        if timestamp_column is not None:
            working[timestamp_column] = pd.to_datetime(
                working[timestamp_column], errors="coerce"
            )
            working = working.dropna(subset=[timestamp_column])
            working.set_index(timestamp_column, inplace=True)
        elif "date" in column_lookup and "time" in column_lookup:
            date_col = column_lookup["date"]
            time_col = column_lookup["time"]
            combined = (
                working[date_col].astype(str).str.strip()
                + " "
                + working[time_col].astype(str).str.strip()
            )
            working.index = pd.to_datetime(combined, errors="coerce")
            working.drop(columns=[date_col, time_col], inplace=True)
            working = working[working.index.notna()]
        elif working.index.name is not None:
            working.index = pd.to_datetime(working.index, errors="coerce")
            working = working[working.index.notna()]
        else:
            raise RuntimeError(f"No timestamp column detected for {source}")
        working.sort_index(inplace=True)
        if working.empty:
            return working
        LOGGER.info(
            "Condition met: normalise_history_frame_complete",
            extra={
                "event": "normalise_history_frame_complete",
                "rows": len(working.index),
                "source": str(source),
            },
        )
        return working
    except Exception as exc:  # noqa: BLE001 - defensive logging
        LOGGER.error(
            "Failure in _normalise_history_frame: %s",
            exc,
            extra={"event": "normalise_history_frame_error", "source": str(source)},
        )
        raise RuntimeError("Unable to normalise history frame") from exc


def _load_market_data(symbol: str, days: int, base_dir: Path) -> pd.DataFrame:
    """Load OHLCV market data for the most recent *days* calendar days.

    Args:
        symbol: Trading symbol to load.
        days: Number of calendar days to include.
        base_dir: Directory containing the historical files.

    Returns:
        pd.DataFrame: Combined dataframe indexed by timestamp.

    Raises:
        FileNotFoundError: Propagated if no data is available.
        RuntimeError: If parsing fails unexpectedly.
    """

    LOGGER.debug(
        "Entered _load_market_data",
        extra={"event": "load_market_data_enter", "symbol": symbol, "days": days},
    )
    try:
        files = _discover_history_files(symbol, base_dir)
        symbol_upper = symbol.upper()
        frames: list[pd.DataFrame] = []
        for path in files:
            LOGGER.debug(
                "Loading history file for quick backtest",
                extra={
                    "event": "load_market_data_file",
                    "symbol": symbol_upper,
                    "path": str(path),
                },
            )
            if path.suffix.lower() == ".csv":
                raw_frame = pd.read_csv(path)
            else:
                raw_frame = pd.read_parquet(path)
            normalised = _normalise_history_frame(
                raw_frame, symbol=symbol_upper, source=path
            )
            if normalised.empty:
                LOGGER.debug(
                    "Skipping history frame with no matching rows",
                    extra={
                        "event": "load_market_data_skip_empty",
                        "symbol": symbol_upper,
                        "path": str(path),
                    },
                )
                continue
            frames.append(normalised)
        if not frames:
            raise FileNotFoundError(f"No usable history for {symbol_upper}")
        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")]
        if combined.index.tz is None:
            tz = _INDIA_TZ if _INDIA_TZ is not None else timezone.utc
            combined.index = combined.index.tz_localize(
                tz, nonexistent="shift_forward", ambiguous="NaT"
            )
        combined = combined[~combined.index.isna()]
        combined.index = combined.index.tz_convert(timezone.utc)
        combined = combined.sort_index()
        cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=days)
        recent = combined[combined.index >= cutoff]
        if recent.empty:
            raise FileNotFoundError(
                f"No market data found for {symbol_upper} within the last {days} days"
            )
        if "close" not in recent.columns:
            raise RuntimeError("Missing required column: close")
        LOGGER.info(
            "Condition met: load_market_data_complete",
            extra={
                "event": "load_market_data_complete",
                "rows": len(recent),
                "symbol": symbol_upper,
            },
        )
        return recent
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - defensive logging
        LOGGER.error(
            "Failure in _load_market_data: %s",
            exc,
            extra={"event": "load_market_data_error"},
        )
        raise RuntimeError("Failed to load market data") from exc


def run_quick_backtest(
    symbol: str, days: int, *, data_directory: Path | None = None
) -> BacktestSummary:
    """Run a lightweight RSI backtest for *symbol* over *days*.

    Args:
        symbol: Instrument symbol (case-insensitive).
        days: Number of calendar days to replay.
        data_directory: Optional override for history root directory.

    Returns:
        BacktestSummary: Aggregated result payload.

    Raises:
        FileNotFoundError: If no data is available.
        RuntimeError: If the simulation fails.
    """

    LOGGER.debug(
        "Entered run_quick_backtest",
        extra={"event": "run_quick_backtest_enter", "symbol": symbol, "days": days},
    )
    try:
        history_dir = _resolve_backtest_history_root(data_directory)
        LOGGER.info(
            "Condition met: run_quick_backtest_history_resolved",
            extra={
                "event": "run_quick_backtest_history_resolved",
                "symbol": symbol,
                "days": days,
                "history_dir": str(history_dir),
            },
        )
        data = _load_market_data(symbol, days, history_dir)
        strategy = _RSIMeanReversionBridge(symbol)
        config = BacktestConfig(generate_visualizations=False)
        engine = BacktestEngine(data, strategy, config)
        result = engine.run()
        total_pnl = float(sum(trade.pnl for trade in result.trades))
        trade_count = len(result.trades)
        win_rate = float(result.performance.get("win_rate", 0.0) * 100.0)
        max_drawdown = float(abs(result.performance.get("max_drawdown", 0.0) * 100.0))
        avg_trade_pnl = float(total_pnl / trade_count) if trade_count else 0.0
        breakdown = {
            strategy.name: {
                "pnl": total_pnl,
                "trades": float(trade_count),
                "win_rate": win_rate,
            }
        }
        summary = BacktestSummary(
            symbol=symbol.upper(),
            days=days,
            total_pnl=total_pnl,
            trade_count=trade_count,
            avg_trade_pnl=avg_trade_pnl,
            win_rate_pct=win_rate,
            max_drawdown_pct=max_drawdown,
            strategy_breakdown=breakdown,
            result=result,
        )
        LOGGER.info(
            "Condition met: run_quick_backtest_complete",
            extra={
                "event": "run_quick_backtest_complete",
                "symbol": symbol,
                "days": days,
                "trades": trade_count,
                "pnl": total_pnl,
            },
        )
        return summary
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - defensive logging
        LOGGER.error(
            "Failure in run_quick_backtest: %s",
            exc,
            extra={"event": "run_quick_backtest_error"},
        )
        raise RuntimeError("Quick backtest failed") from exc
