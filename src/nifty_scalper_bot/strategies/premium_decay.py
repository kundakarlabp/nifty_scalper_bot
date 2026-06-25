"""Premium decay strategy capturing theta via short strangles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from datetime import datetime, time, timezone
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)
from zoneinfo import ZoneInfo

from nifty_scalper_bot.execution.order_manager import OrderType
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.risk.risk_manager import OrderSignal
from nifty_scalper_bot.strategies.signal_generator import Signal
from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)


@runtime_checkable
class OrderManagerProtocol(Protocol):
    """Protocol describing the order manager operations used by the strategy."""

    def place_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        product: str | None = None,
        tag: str | None = None,
        trailing_spec: Any | None = None,
        max_slippage_per_lot: float | None = None,
    ) -> str:
        """Submit an order and return the broker identifier."""


@runtime_checkable
class RiskManagerProtocol(Protocol):
    """Protocol describing the risk manager interaction surface."""

    def check_order(self, signal: OrderSignal, live_enabled: bool) -> tuple[bool, str]:
        """Return whether the provided order signal is allowed to proceed."""


@runtime_checkable
class StrategyOrchestratorProtocol(Protocol):
    """Protocol describing orchestration hooks for correlation and capital gates."""

    def filter_signal(
        self,
        signal: Signal,
        indicators: Mapping[str, Any],
        position_manager: PositionManagerProtocol,
    ) -> Signal | None:
        """Return filtered signal or ``None`` if the leg must be skipped."""

    def notify_submission(self, signal: Signal, underlying: str) -> None:
        """Record that a leg has been submitted to the broker."""

    def notify_exit(self, underlying: str) -> None:
        """Record that the strategy squared off the provided underlying."""


@runtime_checkable
class PositionManagerProtocol(Protocol):
    """Protocol describing the position manager dependency."""

    def get_all_positions(self) -> Sequence[Any]:
        """Return the current set of broker positions."""


Clock = Callable[[], datetime]


@dataclass(slots=True)
class ShortLegState:
    """Track individual short option leg metadata for monitoring."""

    symbol: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    order_id: str


@dataclass(slots=True)
class ShortStrangleState:
    """State container for the active short strangle position."""

    underlying: str
    ce: ShortLegState
    pe: ShortLegState
    opened_at: datetime
    iv_entry: float


class PremiumDecayStrategy:
    """Backtest/paper short-strangle strategy; never a live execution authority."""

    LIVE_CAPABLE = False

    def __init__(
        self,
        *,
        order_manager: OrderManagerProtocol,
        risk_manager: RiskManagerProtocol,
        orchestrator: StrategyOrchestratorProtocol,
        position_manager: PositionManagerProtocol,
        atr_threshold: float = 25.0,
        adx_threshold: float = 18.0,
        iv_exit_threshold: float = 0.40,
        lots: int = 1,
        max_slippage_per_lot: float = 15.0,
        tag: str = "premium_decay",
        square_off_time: time = time(hour=15, minute=10),
        clock: Clock | None = None,
    ) -> None:
        """Initialise the strategy with execution and risk dependencies.

        Args:
            order_manager: Order routing component.
            risk_manager: Risk engine used for pre-trade checks.
            orchestrator: Strategy orchestrator for capital/correlation gates.
            position_manager: Position manager reference for orchestrator context.
            atr_threshold: Maximum ATR allowed to consider market range-bound.
            adx_threshold: Maximum ADX allowed to consider market range-bound.
            iv_exit_threshold: IV level that triggers defensive exit.
            lots: Default lot count when entering a strangle.
            max_slippage_per_lot: Maximum adverse slippage permitted per lot in INR.
            tag: Tag applied to broker orders for auditability.
            square_off_time: Time-of-day (IST) to force square-off of open legs.
            clock: Optional callable returning current :class:`datetime`.

        Raises:
            ValueError: If configured ``lots`` is not positive.
        """

        if lots <= 0:
            raise ValueError("lots must be positive")
        self._order_manager = order_manager
        self._risk_manager = risk_manager
        self._orchestrator = orchestrator
        self._position_manager = position_manager
        self._atr_threshold = float(max(atr_threshold, 0.0))
        self._adx_threshold = float(max(adx_threshold, 0.0))
        self._iv_exit_threshold = float(max(iv_exit_threshold, 0.0))
        self._lots = int(lots)
        self._max_slippage_per_lot = float(max_slippage_per_lot)
        self._tag = tag
        self._square_off_time = square_off_time
        self._clock: Clock = (
            clock if clock is not None else (lambda: datetime.now(timezone.utc))
        )
        self._logger = LOGGER
        self._tz = ZoneInfo("Asia/Kolkata")
        self._active: ShortStrangleState | None = None
        self._name = "PremiumDecay"

    def evaluate_entry(
        self,
        *,
        underlying: str,
        indicators: Mapping[str, Any],
        option_chain: Sequence[SelectedContract],
        iv: float,
    ) -> bool:
        """Attempt to open a short strangle if market conditions permit.

        Args:
            underlying: Underlying symbol used for orchestrator context.
            indicators: Indicator snapshot containing ATR/ADX metrics.
            option_chain: Iterable of option contracts with Greeks metadata.
            iv: Current implied volatility metric for the underlying.

        Returns:
            ``True`` when both legs are submitted successfully, else ``False``.

        Raises:
            None.
        """

        self._logger.debug(
            'Entered PremiumDecayStrategy.evaluate_entry',
            extra={
                'event': 'premium_decay_entry_enter',
                'underlying': underlying,
            },
        )
        execution_mode = str(os.getenv("EXECUTION_MODE", "SHADOW") or "SHADOW").strip().upper()
        live_enabled = execution_mode == "LIVE" or str(
            os.getenv("ENABLE_LIVE", os.getenv("ENABLE_LIVE_TRADING", "false"))
        ).strip().lower() in {"1", "true", "yes", "on"}
        if live_enabled:
            self._logger.error(
                "PREMIUM_DECAY_LIVE_DISABLED reason=noncanonical_multileg_execution",
                extra={
                    "event": "PREMIUM_DECAY_LIVE_DISABLED",
                    "underlying": underlying,
                    "reason": "noncanonical_multileg_execution",
                },
            )
            return False
        try:
            if self._active is not None:
                self._logger.info(
                    'Condition met: premium_decay_active_position',
                    extra={
                        'event': 'premium_decay_active_position',
                        'underlying': underlying,
                    },
                )
                return False
            atr_value = self._extract_float(indicators, ('atr', 'atr_14'))
            adx_value = self._extract_float(indicators, ('adx', 'adx_14'))
            if atr_value is None or adx_value is None:
                self._logger.info(
                    'Condition met: premium_decay_missing_indicators',
                    extra={
                        'event': 'premium_decay_missing_indicators',
                        'underlying': underlying,
                    },
                )
                return False
            if atr_value > self._atr_threshold or adx_value > self._adx_threshold:
                self._logger.info(
                    'Condition met: premium_decay_trend_filter',
                    extra={
                        'event': 'premium_decay_trend_filter',
                        'underlying': underlying,
                        'atr': atr_value,
                        'adx': adx_value,
                    },
                )
                return False
            iv_floor = min(0.12, self._iv_exit_threshold * 0.6)
            if iv <= 0.0 or iv < iv_floor or iv > self._iv_exit_threshold:
                self._logger.info(
                    'Condition met: premium_decay_iv_filter',
                    extra={
                        'event': 'premium_decay_iv_filter',
                        'underlying': underlying,
                        'iv': iv,
                        'iv_floor': iv_floor,
                        'iv_ceiling': self._iv_exit_threshold,
                    },
                )
                return False
            pair = self._select_contracts(option_chain)
            if pair is None:
                self._logger.info(
                    'Condition met: premium_decay_no_contracts',
                    extra={
                        'event': 'premium_decay_no_contracts',
                        'underlying': underlying,
                    },
                )
                return False
            ce_contract, pe_contract = pair
            lot_size = self._resolve_lot_size(ce_contract, pe_contract)
            if lot_size <= 0:
                return False
            quantity = lot_size * self._lots
            ce_sl, ce_tp = ce_contract.ltp * 2.0, ce_contract.ltp * 0.5
            pe_sl, pe_tp = pe_contract.ltp * 2.0, pe_contract.ltp * 0.5
            ce_signal = Signal(
                action='SELL',
                symbol=ce_contract.symbol,
                quantity=quantity,
                confidence=0.55,
                reason='Premium decay short call',
                stop_loss=ce_sl,
                take_profit=ce_tp,
                metadata={'strategy': self._name, 'underlying': underlying},
            )
            pe_signal = Signal(
                action='SELL',
                symbol=pe_contract.symbol,
                quantity=quantity,
                confidence=0.55,
                reason='Premium decay short put',
                stop_loss=pe_sl,
                take_profit=pe_tp,
                metadata={'strategy': self._name, 'underlying': underlying},
            )
            if not self._risk_allows(
                ce_contract.symbol, 'SELL', quantity, ce_contract.ltp, ce_sl, ce_tp
            ):
                return False
            if not self._risk_allows(
                pe_contract.symbol, 'SELL', quantity, pe_contract.ltp, pe_sl, pe_tp
            ):
                return False
            ce_filtered = self._orchestrator.filter_signal(
                ce_signal, indicators, self._position_manager
            )
            if ce_filtered is None:
                self._logger.info(
                    'Condition met: premium_decay_orchestrator_block',
                    extra={
                        'event': 'premium_decay_orchestrator_block',
                        'underlying': underlying,
                        'leg': 'CE',
                    },
                )
                return False
            pe_filtered = self._orchestrator.filter_signal(
                pe_signal, indicators, self._position_manager
            )
            if pe_filtered is None:
                self._logger.info(
                    'Condition met: premium_decay_orchestrator_block',
                    extra={
                        'event': 'premium_decay_orchestrator_block',
                        'underlying': underlying,
                        'leg': 'PE',
                    },
                )
                return False
            ce_order_id = self._submit_order(
                symbol=ce_contract.symbol,
                price=ce_contract.ltp,
                quantity=quantity,
            )
            if not ce_order_id:
                return False
            pe_order_id = self._submit_order(
                symbol=pe_contract.symbol,
                price=pe_contract.ltp,
                quantity=quantity,
            )
            if not pe_order_id:
                self._logger.error(
                    'Failure in PremiumDecayStrategy.evaluate_entry: second leg rejected',
                    extra={
                        'event': 'premium_decay_second_leg_failed',
                        'underlying': underlying,
                    },
                )
                self._attempt_recover_leg(ce_contract.symbol, quantity)
                return False
            now = self._clock()
            self._active = ShortStrangleState(
                underlying=underlying,
                ce=ShortLegState(
                    symbol=ce_contract.symbol,
                    entry_price=ce_contract.ltp,
                    quantity=quantity,
                    stop_loss=ce_sl,
                    take_profit=ce_tp,
                    order_id=ce_order_id,
                ),
                pe=ShortLegState(
                    symbol=pe_contract.symbol,
                    entry_price=pe_contract.ltp,
                    quantity=quantity,
                    stop_loss=pe_sl,
                    take_profit=pe_tp,
                    order_id=pe_order_id,
                ),
                opened_at=now,
                iv_entry=iv,
            )
            self._orchestrator.notify_submission(ce_filtered, underlying)
            self._orchestrator.notify_submission(pe_filtered, underlying)
            self._logger.info(
                'Condition met: premium_decay_position_opened',
                extra={
                    'event': 'premium_decay_position_opened',
                    'underlying': underlying,
                    'ce': ce_contract.symbol,
                    'pe': pe_contract.symbol,
                    'quantity': quantity,
                    'iv_entry': iv,
                },
            )
            return True
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                'Failure in PremiumDecayStrategy.evaluate_entry: %s',
                exc,
                extra={'event': 'premium_decay_entry_error', 'underlying': underlying},
                exc_info=exc,
            )
            return False

    def evaluate_exit(
        self,
        *,
        option_prices: Mapping[str, float],
        current_iv: float,
        now: datetime | None = None,
    ) -> None:
        """Monitor and exit active strangle based on risk signals.

        Args:
            option_prices: Mapping from option symbol to latest price.
            current_iv: Current implied volatility reading.
            now: Optional timestamp for deterministic testing.

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PremiumDecayStrategy.evaluate_exit",
            extra={"event": "premium_decay_exit_enter"},
        )
        state = self._active
        if state is None:
            return
        moment = now or self._clock()
        if current_iv > self._iv_exit_threshold:
            self._logger.info(
                "Condition met: premium_decay_iv_spike",
                extra={
                    "event": "premium_decay_iv_spike",
                    "underlying": state.underlying,
                    "iv": current_iv,
                },
            )
            self._close_position(option_prices, moment, reason="iv_spike")
            return
        if self._should_square_off(moment):
            self._logger.info(
                "Condition met: premium_decay_square_off",
                extra={
                    "event": "premium_decay_square_off",
                    "underlying": state.underlying,
                },
            )
            self._close_position(option_prices, moment, reason="square_off")
            return
        ce_price = option_prices.get(state.ce.symbol)
        pe_price = option_prices.get(state.pe.symbol)
        if ce_price is not None and ce_price >= state.ce.stop_loss:
            self._logger.info(
                "Condition met: premium_decay_ce_stop",
                extra={
                    "event": "premium_decay_ce_stop",
                    "underlying": state.underlying,
                    "price": ce_price,
                },
            )
            self._close_position(option_prices, moment, reason="ce_stop")
            return
        if pe_price is not None and pe_price >= state.pe.stop_loss:
            self._logger.info(
                "Condition met: premium_decay_pe_stop",
                extra={
                    "event": "premium_decay_pe_stop",
                    "underlying": state.underlying,
                    "price": pe_price,
                },
            )
            self._close_position(option_prices, moment, reason="pe_stop")
            return
        ce_tp_hit = ce_price is not None and ce_price <= state.ce.take_profit
        pe_tp_hit = pe_price is not None and pe_price <= state.pe.take_profit
        if ce_tp_hit and pe_tp_hit:
            self._logger.info(
                "Condition met: premium_decay_target",
                extra={
                    "event": "premium_decay_target",
                    "underlying": state.underlying,
                },
            )
            self._close_position(option_prices, moment, reason="target")

    def _select_contracts(
        self, option_chain: Sequence[SelectedContract]
    ) -> tuple[SelectedContract, SelectedContract] | None:
        """Select CE and PE contracts closest to the desired delta band.

        Args:
            option_chain: Sequence of candidate contracts with Greeks data.

        Returns:
            Tuple containing the preferred CE and PE contracts, otherwise ``None``.

        Raises:
            None.
        """

        ce_candidate: SelectedContract | None = None
        pe_candidate: SelectedContract | None = None
        best_ce_score = float("inf")
        best_pe_score = float("inf")
        for contract in option_chain:
            option_type = getattr(contract, "option_type", "").upper()
            delta = getattr(contract, "delta", None)
            ltp = getattr(contract, "ltp", 0.0)
            if ltp <= 0 or delta is None:
                continue
            score = abs(abs(float(delta)) - 0.25)
            if option_type == "CE" and score < best_ce_score:
                ce_candidate = contract
                best_ce_score = score
            elif option_type == "PE" and score < best_pe_score:
                pe_candidate = contract
                best_pe_score = score
        if ce_candidate is None or pe_candidate is None:
            return None
        return ce_candidate, pe_candidate

    def _resolve_lot_size(self, ce: SelectedContract, pe: SelectedContract) -> int:
        """Determine the lot size ensuring both legs share identical multipliers.

        Args:
            ce: Selected call contract.
            pe: Selected put contract.

        Returns:
            Lot size when valid metadata exists, otherwise ``0``.

        Raises:
            None.
        """

        lot_size_ce = self._extract_lot_size(ce)
        lot_size_pe = self._extract_lot_size(pe)
        if lot_size_ce <= 0 or lot_size_pe <= 0 or lot_size_ce != lot_size_pe:
            self._logger.error(
                "Failure in PremiumDecayStrategy._resolve_lot_size: invalid lot data",
                extra={
                    "event": "premium_decay_lot_failure",
                    "ce_lot": lot_size_ce,
                    "pe_lot": lot_size_pe,
                },
            )
            return 0
        return lot_size_ce

    def _risk_allows(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        price: float,
        stop_loss: float,
        take_profit: float,
    ) -> bool:
        """Run risk checks for the proposed leg before routing orders.

        Args:
            symbol: Contract identifier being evaluated.
            side: Side of the order being submitted.
            quantity: Quantity of the order.
            price: Reference price for the risk engine.
            stop_loss: Configured stop-loss for the leg.
            take_profit: Configured take-profit for the leg.

        Returns:
            ``True`` when risk approves the leg or no hook is available.

        Raises:
            None.
        """

        if not hasattr(self._risk_manager, "check_order"):
            return True
        try:
            signal = OrderSignal(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            allowed, reason = self._risk_manager.check_order(signal, live_enabled=True)
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PremiumDecayStrategy._risk_allows: %s",
                exc,
                extra={"event": "premium_decay_risk_error", "symbol": symbol},
            )
            return False
        if not allowed:
            self._logger.info(
                "Condition met: premium_decay_risk_block",
                extra={
                    "event": "premium_decay_risk_block",
                    "symbol": symbol,
                    "reason": reason,
                },
            )
            return False
        return True

    def _submit_order(self, *, symbol: str, price: float, quantity: int) -> str:
        """Submit a SELL order for the provided contract using the order manager.

        Args:
            symbol: Contract identifier being sold.
            price: Limit price used for routing.
            quantity: Quantity in units.

        Returns:
            Broker order identifier if submission succeeds, else empty string.

        Raises:
            None.
        """

        try:
            return self._order_manager.place_order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=price,
                product="MIS",
                tag=self._tag,
                max_slippage_per_lot=self._max_slippage_per_lot,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PremiumDecayStrategy._submit_order: %s",
                exc,
                extra={"event": "premium_decay_submit_error", "symbol": symbol},
            )
            return ""

    def _attempt_recover_leg(self, symbol: str, quantity: int) -> None:
        """Attempt to neutralise a single executed leg if the pair fails.

        Args:
            symbol: Contract identifier that must be bought back.
            quantity: Quantity to cover.

        Returns:
            None.

        Raises:
            None.
        """

        try:
            self._order_manager.place_order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=None,
                product="MIS",
                tag=f"{self._tag}_recover",
                max_slippage_per_lot=self._max_slippage_per_lot,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "Failure in PremiumDecayStrategy._attempt_recover_leg: %s",
                exc,
                extra={
                    "event": "premium_decay_recover_failed",
                    "symbol": symbol,
                },
            )

    def _close_position(
        self,
        option_prices: Mapping[str, float],
        moment: datetime,
        *,
        reason: str,
    ) -> None:
        """Exit all active legs and notify orchestrator of the closure.

        Args:
            option_prices: Latest option prices used for exit routing.
            moment: Timestamp of the closure.
            reason: Reason for the exit (e.g., ``iv_spike``).

        Returns:
            None.

        Raises:
            None.
        """

        self._logger.debug(
            "Entered PremiumDecayStrategy._close_position",
            extra={
                "event": "premium_decay_close_enter",
                "reason": reason,
            },
        )
        state = self._active
        if state is None:
            return
        for leg in (state.ce, state.pe):
            target_price = option_prices.get(leg.symbol, leg.entry_price)
            try:
                self._order_manager.place_order(
                    symbol=leg.symbol,
                    side="BUY",
                    quantity=leg.quantity,
                    order_type=OrderType.LIMIT,
                    price=target_price,
                    product="MIS",
                    tag=f"{self._tag}_{reason}",
                    max_slippage_per_lot=self._max_slippage_per_lot,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Failure in PremiumDecayStrategy._close_position: %s",
                    exc,
                    extra={
                        "event": "premium_decay_exit_failed",
                        "symbol": leg.symbol,
                    },
                )
        self._orchestrator.notify_exit(state.underlying)
        self._active = None
        self._logger.info(
            "Condition met: premium_decay_position_closed",
            extra={
                "event": "premium_decay_position_closed",
                "underlying": state.underlying,
                "reason": reason,
                "timestamp": moment.isoformat(),
            },
        )

    def _should_square_off(self, moment: datetime) -> bool:
        """Return ``True`` when time-based square-off should be triggered.

        Args:
            moment: Current timestamp in UTC.

        Returns:
            ``True`` if the localised timestamp is past the configured cutoff.

        Raises:
            None.
        """

        local = moment.astimezone(self._tz)
        cutoff = local.replace(
            hour=self._square_off_time.hour,
            minute=self._square_off_time.minute,
            second=0,
            microsecond=0,
        )
        return local >= cutoff

    @staticmethod
    def _extract_float(payload: Mapping[str, Any], keys: Iterable[str]) -> float | None:
        """Extract a float from the mapping using the first matching key.

        Args:
            payload: Mapping containing indicator values.
            keys: Candidate keys to inspect.

        Returns:
            Floating value if found and coercible, else ``None``.

        Raises:
            None.
        """

        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_lot_size(contract: SelectedContract) -> int:
        """Read the lot size metadata from the contract when available.

        Args:
            contract: Option contract containing metadata dictionary.

        Returns:
            Integer lot size if present and valid, otherwise ``0``.

        Raises:
            None.
        """

        metadata = getattr(contract, "metadata", None)
        if isinstance(metadata, Mapping):
            lot_value = metadata.get("lot_size") or metadata.get("lotSize")
            if lot_value is not None:
                try:
                    lot_size = int(float(lot_value))
                    if lot_size > 0:
                        return lot_size
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return 0
        return 0


__all__ = ["PremiumDecayStrategy"]
