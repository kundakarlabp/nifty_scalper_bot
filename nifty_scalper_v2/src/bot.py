"""NiftyScalperBot v2 — Top-level orchestrator.

Implements the 18-step startup sequence from MASTER_REWRITE_SPEC.md Section 6.1.
Main event loop runs at 100ms cadence (Section 6.2).
Signal filtering per Section 6.3.
"""

from __future__ import annotations

import asyncio
import time
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


class NiftyScalperBot:
    """Top-level orchestrator. One instance per process."""

    VERSION = "2.0.0"

    def __init__(self) -> None:
        from .config.settings import get_settings
        self._settings = get_settings()
        self._running = False
        self._started_at: Optional[datetime] = None

        # Components (wired in startup())
        self._broker = None
        self._tick_hub = None
        self._candle_engine = None
        self._indicator_engine = None
        self._option_universe = None
        self._streaming = None
        self._risk_manager = None
        self._session_gate = None
        self._regime_detector = None
        self._strategy_manager = None
        self._strategy_runner = None
        self._bracket_manager = None
        self._lifecycle_manager = None
        self._execution_router = None
        self._reconciler = None
        self._db = None
        self._trade_log = None
        self._journal_queries = None
        self._notifier = None
        self._telegram_bot = None
        self._scheduler = None
        self._metrics = None

    async def startup(self) -> None:
        """18-step startup sequence. Any fatal step raises and exits."""
        log.info("NiftyScalperBot v%s starting...", self.VERSION)
        s = self._settings

        # Step 1: Logging already configured in main.py

        # Step 2: Database
        from .journal.db import AsyncDB
        from .journal.trade_log import TradeLog
        from .journal.queries import JournalQueries
        self._db = AsyncDB(s.infra.db_path)
        await self._db.init()
        self._trade_log = TradeLog(self._db)
        self._journal_queries = JournalQueries(self._db)
        log.info("Database initialized: %s", s.infra.db_path)

        # Step 3: Metrics
        from .infra.metrics import METRICS
        self._metrics = METRICS
        try:
            METRICS.start_server(s.infra.metrics_port)
            log.info("Prometheus metrics on :%d", s.infra.metrics_port)
        except Exception as e:
            log.warning("Metrics server failed (already running?): %s", e)

        # Step 4: Broker
        mode = s.trading.execution_mode.value
        if mode == "PAPER":
            from .broker.paper import PaperBroker
            self._broker = PaperBroker(s.execution.slippage_model)
            log.info("Paper broker initialized")
        else:
            from .broker.zerodha import ZerodhaBroker
            self._broker = ZerodhaBroker(s.broker)
            await self._broker.connect()
            log.info("Zerodha broker connected")

        # Step 5: Tick hub
        from .data.tick_hub import TickHub
        self._tick_hub = TickHub(warmup_threshold=s.streaming.warmup_ticks)

        # Step 6: Candle engine
        from .data.candle_engine import CandleEngine
        self._candle_engine = CandleEngine(self._tick_hub.event_bus)

        # Step 7: Indicator engine
        from .indicators.engine import IndicatorEngine
        self._indicator_engine = IndicatorEngine(
            self._tick_hub.event_bus,
            warmup_candles=s.streaming.warmup_candles,
        )

        # Step 8: Option universe
        from .options.universe import OptionUniverse
        self._option_universe = OptionUniverse(self._broker, s.trading.instrument)

        # Step 9: Risk manager
        from_daily = await self._journal_queries.get_daily_pnl()
        today_count = await self._journal_queries.get_trade_count_today()
        from .risk.manager import RiskManager
        self._risk_manager = RiskManager(
            capital=s.risk.capital,
            daily_loss_pct=s.risk.daily_loss_pct,
            max_daily_trades=s.risk.max_daily_trades,
            max_concurrent=s.risk.max_concurrent_positions,
            consecutive_loss_cooldown=s.risk.consecutive_loss_cooldown,
            cooldown_minutes=s.risk.cooldown_minutes,
            vix_extreme_threshold=s.risk.vix_extreme_threshold,
        )

        # Step 10: Session gate
        from .risk.session import SessionGate
        self._session_gate = SessionGate()

        # Step 11: Regime detector
        from .regime.detector import RegimeDetector
        self._regime_detector = RegimeDetector()

        # Step 12: Strategies
        from .strategies.manager import StrategyManager
        from .strategies.runner import StrategyRunner
        from .strategies.impl.vwap_pro import VWAPProStrategy
        from .strategies.impl.smc_liquidity import SMCLiquidityStrategy
        from .strategies.impl.rsi_divergence import RSIDivergenceStrategy
        from .strategies.impl.gamma_scalping import GammaScalpingStrategy
        from .strategies.impl.oi_max_pain import OIMaxPainStrategy
        from .strategies.impl.orb_pro import ORBProStrategy
        from .strategies.impl.bb_squeeze import BBSqueezeStrategy
        from .strategies.impl.cpr_breakout import CPRBreakoutStrategy
        from .strategies.impl.order_flow import OrderFlowStrategy
        from .strategies.impl.straddle_theta import StraddleThetaStrategy
        from .strategies.impl.trend_momentum import TrendMomentumStrategy
        from .strategies.impl.tuesday_gamma import TuesdayGammaStrategy

        ss = s.strategies
        strategies = [
            VWAPProStrategy(ss.vwap_pro, self._option_universe),
            SMCLiquidityStrategy(ss.smc_liquidity),
            RSIDivergenceStrategy(ss.rsi_divergence),
            GammaScalpingStrategy(ss.gamma_scalping),
            OIMaxPainStrategy(ss.oi_max_pain),
            ORBProStrategy(ss.orb_pro),
            BBSqueezeStrategy(ss.bb_squeeze),
            CPRBreakoutStrategy(ss.cpr_breakout),
            OrderFlowStrategy(ss.order_flow),
            StraddleThetaStrategy(ss.straddle_theta),
            TrendMomentumStrategy(ss.trend_momentum),
            TuesdayGammaStrategy(ss.tuesday_gamma),
        ]
        self._strategy_manager = StrategyManager(strategies)
        self._strategy_runner = StrategyRunner(self._strategy_manager, self._risk_manager)

        # Step 13: Execution
        from .execution.bracket import BracketManager
        from .execution.lifecycle import LifecycleManager
        from .execution.router import ExecutionRouter
        from .execution.reconciler import AsyncReconciler
        self._bracket_manager = BracketManager()
        self._lifecycle_manager = LifecycleManager(self._bracket_manager, s.lifecycle)
        self._execution_router = ExecutionRouter(
            self._broker, s.trading, self._bracket_manager
        )
        self._reconciler = AsyncReconciler(self._broker, s.execution.reconcile_interval_sec)
        await self._reconciler.start()

        # Step 14: Telegram / Notifier
        from .notifications.telegram.bot import TelegramBot
        from .notifications.notifier import Notifier
        self._telegram_bot = TelegramBot(s.telegram)
        await self._telegram_bot.setup()
        self._notifier = Notifier(self._telegram_bot)
        self._execution_router._notifier = self._notifier

        # Step 15: Scheduler
        from .infra.scheduler import BotScheduler
        self._scheduler = BotScheduler()
        self._scheduler.schedule_daily_summary(
            s.infra.daily_summary_time, self._send_daily_summary
        )
        self._scheduler.schedule_session_reset(self._reset_daily)
        self._scheduler.start()

        # Step 16: Streaming
        token_map = {}  # In production: populated from instrument resolver
        if not s.streaming.websocket_disabled:
            from .streaming.websocket_stream import WebSocketStream
            self._streaming = WebSocketStream(
                self._tick_hub, s.broker, s.streaming, token_map
            )
            try:
                self._streaming.start()
            except Exception as e:
                log.warning("WebSocket failed, falling back to polling: %s", e)
                self._streaming = None

        if self._streaming is None:
            from .streaming.polling_stream import PollingStream
            self._streaming = PollingStream(
                self._tick_hub, self._broker, s.streaming, token_map
            )
            await self._streaming.start()

        # Step 17: Telegram bot start
        await self._telegram_bot.start()

        # Step 18: READY
        self._started_at = datetime.now(timezone.utc)
        self._running = True
        await self._notifier.send_text(
            f"🚀 *NiftyScalperBot v{self.VERSION} READY*\n"
            f"Mode: {mode} | Capital: ₹{s.risk.capital:,.0f}"
        )
        log.info("Bot READY in %s mode", mode)

    async def run(self) -> None:
        """Start the main 100ms event loop."""
        await self.startup()
        try:
            await self._run_loop()
        finally:
            await self.shutdown()

    async def _run_loop(self) -> None:
        """Main 100ms event loop per spec Section 6.2."""
        log.info("Main loop starting (100ms cadence)")
        while self._running:
            loop_start = time.monotonic()
            try:
                # 1. Session gate
                if not self._session_gate.is_trading_now():
                    await asyncio.sleep(1.0)
                    continue

                # 2. Warmup guard
                if not self._tick_hub.warmup_complete:
                    await asyncio.sleep(0.1)
                    continue

                # 3. Get all snapshots
                snapshots = self._indicator_engine.get_all_snapshots()
                if not snapshots:
                    await asyncio.sleep(0.1)
                    continue

                # 4. Regime detection (cached 5 min)
                # Use the first valid snapshot for regime
                snap = next((v for v in snapshots.values() if v.is_valid), None)
                if snap is None:
                    await asyncio.sleep(0.1)
                    continue
                regime = self._regime_detector.detect(snap)

                # 5. Get open positions
                open_positions = self._execution_router.get_open_positions()
                self._risk_manager.update_positions(
                    len(open_positions),
                    sum(p.entry_price * p.quantity * 50 for p in open_positions),
                )

                # 6. Generate signals
                signals = self._strategy_runner.run(snapshots, regime, open_positions)

                # 7. Execute signals
                for signal in signals:
                    await self._execution_router.route(signal)
                    if self._metrics:
                        self._metrics.signals_generated.labels(
                            strategy=signal.strategy_name,
                            direction=signal.direction.value,
                        ).inc()

                # 8. Lifecycle evaluation
                exits = self._lifecycle_manager.evaluate_all(snapshots)
                for exit_sig in exits:
                    await self._execution_router.handle_exit(exit_sig)

                # 9. Metrics
                if self._metrics:
                    self._metrics.open_positions.set(len(open_positions))
                    self._metrics.loop_duration.observe(time.monotonic() - loop_start)

            except Exception as exc:
                log.error("Loop error: %s", exc, exc_info=True)
                if self._metrics:
                    self._metrics.loop_errors.inc()

            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, 0.1 - elapsed)
            await asyncio.sleep(sleep_time)

    async def shutdown(self) -> None:
        """Graceful teardown."""
        log.info("Shutting down...")
        self._running = False

        if self._reconciler:
            await self._reconciler.stop()
        if hasattr(self._streaming, "stop"):
            if asyncio.iscoroutinefunction(self._streaming.stop):
                await self._streaming.stop()
            else:
                self._streaming.stop()
        if self._telegram_bot:
            await self._telegram_bot.stop()
        if self._scheduler:
            self._scheduler.stop()

        log.info("NiftyScalperBot shutdown complete")

    async def _send_daily_summary(self) -> None:
        if not self._journal_queries or not self._notifier:
            return
        pnl = await self._journal_queries.get_daily_pnl()
        trades = await self._journal_queries.get_trade_count_today()
        win_rate = await self._journal_queries.get_win_rate(days=1)
        max_dd = await self._journal_queries.get_max_drawdown(days=1)
        await self._notifier.send_daily_summary({
            "trades": trades, "net_pnl": pnl,
            "win_rate": win_rate, "max_drawdown": max_dd,
        })

    async def _reset_daily(self) -> None:
        if self._risk_manager:
            self._risk_manager.reset_daily()
        if self._strategy_manager:
            self._strategy_manager.reset_all_daily()
        log.info("Daily counters reset")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        return (datetime.now(timezone.utc) - self._started_at).total_seconds()

    @property
    def mode(self) -> str:
        return self._settings.trading.execution_mode.value
