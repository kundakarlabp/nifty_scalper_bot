"""
Execution Diagnostics Controller:
Handles commands related to order flow, latency, and system integrity.
"""

from __future__ import annotations
import typing as t
from datetime import datetime, timezone

from nifty_scalper_bot.utils.response_builder import EMOJI
from telegram.ext import ContextTypes, Update
from nifty_scalper_bot.utils.pricing import canonical_price_source

if t.TYPE_CHECKING:
    from nifty_scalper_bot.notifications.telegram_controller import TelegramBot # Import for type hinting

# Define a minimal set of dependencies required for these diagnostics
@t.dataclass(slots=True)
class ExecutionControllerDeps:
    fetch_metric_summary: t.Callable[[], t.Mapping[str, t.Any]]
    collect_recent_errors: t.Callable[..., t.Dict[str, int]]
    reconcile_status_line: t.Callable[[], t.Tuple[t.Optional[str], t.Optional[str]]]
    reconcile_last_success_at: t.Optional[datetime]
    reconcile_last_failure_at: t.Optional[datetime]
    reconcile_alert_failures: int
    response_builder: t.Any
    coerce_float_value: t.Any
    format_currency: t.Any

class ExecutionDiagnosticsController:
    """Controller for execution-specific diagnostic commands."""

    def __init__(self, deps: ExecutionControllerDeps) -> None:
        self.deps = deps
        self.rb = deps.response_builder
        self._coerce_float_value = deps.coerce_float_value
        self._format_currency = deps.format_currency

    @t.no_type_check
    async def cmd_rejections(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Lists recent order rejections by reason and count."""
        # Note: _guard is handled by the caller (TelegramBot) for security, but we perform it conceptually here
        # chat = await self._guard(update) 
        # if chat is None: return

        chat = update.effective_chat # Assuming update is validated and chat is present

        try:
            # Get raw error map from the shared metric logic
            rejection_map = self.deps.collect_recent_errors(limit=20)
            
            # Filter specifically for rejection-like reasons
            filtered_rejections = {
                k: v for k, v in rejection_map.items() 
                if 'reject' in k.lower() or 'block' in k.lower() or 'margin' in k.lower() or 'failure' in k.lower()
            }
            
            if not filtered_rejections:
                await chat.send_message("No recent order rejections or critical failures recorded in metrics (last 5m).")
                return

            lines: list[str] = [f"{self.rb.section('Recent Order Rejections')} {EMOJI['fail']}", ]
            
            for reason, count in sorted(filtered_rejections.items(), key=lambda item: item[1], reverse=True):
                lines.append(f"❌ <b>{self.rb.esc(reason)}</b>: {count} times")
            
            lines.append(f"\n{EMOJI['hint']} **Hint**: Review detailed logs or broker status for fatal errors (e.g., Auth, Margin).")

            await chat.send_message(self.rb.br().join(lines), parse_mode="HTML", disable_web_page_preview=True)
            
        except Exception as exc:
            # Fallback reply is handled by TelegramBot wrapper for safety, but log here for diagnosis
            await chat.send_message("Failed to retrieve rejection data due to an internal error.")

    @t.no_type_check
    async def cmd_latencies(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Summary of order and tick processing latency."""
        chat = update.effective_chat 

        summary_data = self.deps.fetch_metric_summary()
        
        # Use explicit key lookup based on metric summary payload
        tick_p95 = self._coerce_float_value(summary_data.get("tick_latency_p95"), field="tick_p95")
        order_p95 = self._coerce_float_value(summary_data.get("order_latency_p95"), field="order_p95")
        max_staleness = self._coerce_float_value(summary_data.get("max_tick_staleness"), field="max_staleness")
        
        lines: list[str] = [f"{self.rb.section('System Latencies')} {EMOJI['clock']}", ]

        def _fmt_seconds(value: float | None) -> str:
            if value is None: return "n/a"
            return f"{value:.3f}s"
        
        lines.append(f"📈 **Tick Latency (p95)**: <code>{_fmt_seconds(tick_p95)}</code>")
        lines.append(f"🧾 **Order Latency (p95)**: <code>{_fmt_seconds(order_p95)}</code>")
        lines.append(f"⏱️ **Max Tick Staleness**: <code>{_fmt_seconds(max_staleness)}</code>")

        if max_staleness and max_staleness > 5.0:
            lines.append(f"{EMOJI['warn']} **Warning**: Max staleness exceeds 5s, check market data connection or processing lag.")
        
        await chat.send_message(self.rb.br().join(lines), parse_mode="HTML", disable_web_page_preview=True)

    @t.no_type_check
    async def cmd_reconcile_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Health and history of position reconciliation."""
        chat = update.effective_chat 
        rb = self.rb
        
        # Get formatted line and hint from dependency's internal state
        reconcile_line, reconcile_hint = self.deps.reconcile_status_line()
        
        lines: list[str] = [f"{rb.section('Reconciliation Health')} {EMOJI['exchange']}", ]
        
        if reconcile_line:
            lines.append(reconcile_line)
        else:
            lines.append("ℹ️ Reconciliation data not yet received or is unavailable.")
        
        last_success_ts = self.deps.reconcile_last_success_at
        if last_success_ts:
            ts_utc = last_success_ts.astimezone(timezone.utc)
            lines.append(f"✅ Last Success: <code>{ts_utc:%H:%M:%S}Z</code>")
            
        last_failure_ts = self.deps.reconcile_last_failure_at
        if last_failure_ts:
            ts_utc = last_failure_ts.astimezone(timezone.utc)
            failure_streak = self.deps.reconcile_alert_failures
            lines.append(f"❌ Last Failure: <code>{ts_utc:%H:%M:%S}Z</code> (Streak: {failure_streak})")

        if reconcile_hint:
            lines.append(f"\n{EMOJI['hint']} **Hint**: {reconcile_hint}")

        # Add metric insight if available
        metrics_summary = self.deps.fetch_metric_summary()
        reconcile_metrics = metrics_summary.get("position_reconcile", {})
        if isinstance(reconcile_metrics, dict):
            failures_5m = int(reconcile_metrics.get("failures_5m", 0) or 0)
            successes_5m = int(reconcile_metrics.get("successes_5m", 0) or 0)
            avg_latency = self._coerce_float_value(reconcile_metrics.get("latency_avg"), field="reco_latency")
            
            if failures_5m > 0 or successes_5m > 0:
                lines.append(f"\n{rb.section('Recent Metrics')}")
                lines.append(f"Successes (5m): {successes_5m}")
                lines.append(f"Failures (5m): {failures_5m}")
                if avg_latency is not None:
                    lines.append(f"Avg Latency: {avg_latency:.3f}s")


        await chat.send_message(self.rb.br().join(lines), parse_mode="HTML", disable_web_page_preview=True)
