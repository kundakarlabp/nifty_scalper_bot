"""Entry-evaluation worker liveness for StrategyRunner.

The runner owns the worker, lock and readiness state; these bound methods read
that state, report its liveness and disarm new entries when the worker stalls.
The methods do not place orders or own an independent copy of runner state.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Mapping


def _entry_eval_liveness_snapshot(
    self: Any, now: float | None = None
) -> dict[str, Any]:
    """Return bounded liveness state for the single entry-eval worker."""
    now = time.monotonic() if now is None else now
    lock = getattr(self, "_eval_gate_lock", None) or threading.Lock()
    with lock:
        started_at = getattr(self, "_entry_eval_active_started_at", None)
        drain_active = bool(getattr(self, "_entry_eval_active", False))
        drain_scheduled = bool(getattr(self, "_entry_eval_drain_scheduled", False))
        pending = sorted(getattr(self, "_pending_entry_eval_symbols", set()))
        active_age = (
            max(0.0, now - float(started_at))
            if drain_active and started_at is not None
            else 0.0
        )
        last_progress = float(
            getattr(
                self,
                "_entry_eval_last_progress_ts",
                getattr(self, "_last_global_eval_ts", now),
            )
            or now
        )
        progress_age = max(0.0, now - last_progress)
        work_outstanding = bool(pending or drain_active or drain_scheduled)
        drain_stranded = bool(pending and not drain_active and not drain_scheduled)
        # A stall that happens before work is queued is invisible to the
        # test below: nothing is pending, no drain is scheduled or active,
        # so continuous ticks with a silent evaluator look identical to a
        # legitimately idle one. Fresh ticks with no evaluation progress is
        # a stall in its own right. Derived, never latched, so it clears as
        # soon as evaluation resumes.
        tick_seen = float(getattr(self, "_last_tick_seen_ts", 0.0) or 0.0)
        selected_tick_seen = float(
            getattr(self, "_last_selected_option_tick_ts", 0.0) or 0.0
        )
        selected_eval_at = float(
            getattr(self, "_last_selected_candidate_eval_completed_ts", 0.0) or 0.0
        )
        selected_eval_age: float | None = (
            max(0.0, now - selected_eval_at) if selected_eval_at > 0.0 else None
        )
        has_selected = bool(
            getattr(self, "_active_selected_ce", None)
            or getattr(self, "_active_selected_pe", None)
        )
        # Context-only drain progress must not mask a dead selected-CE/PE
        # entry path. When selected contracts exist, stall detection keys
        # off selected-option ticks and selected candidate completions.
        if has_selected and selected_tick_seen > 0.0:
            tick_seen = selected_tick_seen
            if selected_eval_at > 0.0:
                last_progress = selected_eval_at
            else:
                last_progress = float(getattr(self, "_runner_started_mono", 0.0) or 0.0)
            progress_age = (
                max(0.0, now - last_progress) if last_progress > 0.0 else progress_age
            )
        tick_age = max(0.0, now - tick_seen) if tick_seen > 0.0 else float("inf")
        dispatch_stall_s = float(
            getattr(self, "_entry_eval_dispatch_stall_s", 120.0) or 120.0
        )
        dispatch_stalled = bool(
            not work_outstanding
            and tick_age <= 5.0
            and progress_age >= dispatch_stall_s
        )
        worker_stalled = (
            bool(
                work_outstanding
                and (active_age >= 90.0 if drain_active else progress_age >= 90.0)
            )
            or dispatch_stalled
        )
        return {
            "tick_age_s": round(tick_age, 1) if tick_age != float("inf") else None,
            "dispatch_stalled": dispatch_stalled,
            "pending_entry_eval": pending,
            "drain_scheduled": drain_scheduled,
            "drain_active": drain_active,
            "drain_active_age_s": round(active_age, 1),
            "last_progress_age_s": round(progress_age, 1),
            "selected_eval_age_s": (
                round(selected_eval_age, 1) if selected_eval_age is not None else None
            ),
            "evaluation_alive": (
                bool(
                    selected_eval_at > 0.0
                    and selected_eval_age is not None
                    and selected_eval_age
                    < float(
                        getattr(self, "_entry_eval_dispatch_stall_s", 120.0) or 120.0
                    )
                )
                if has_selected
                else bool(progress_age < 120.0)
            ),
            "work_outstanding": work_outstanding,
            "drain_stranded": drain_stranded,
            "worker_stalled": worker_stalled,
            "active_symbol": getattr(self, "_entry_eval_active_symbol", None),
            "active_phase": getattr(self, "_entry_eval_active_phase", None),
            "drain_count": int(getattr(self, "_entry_eval_drain_count", 0)),
            "runtime_loop_attached": bool(
                getattr(self, "_runtime_loop_attached", False)
            ),
            "shutdown": bool(getattr(self, "_entry_eval_shutdown", False)),
        }


def _disarm_stalled_entry_worker_if_needed(self: Any, state: Mapping[str, Any]) -> bool:
    """Freeze new entries when the sole worker exceeds its safe age."""
    dispatch_stalled = bool(state.get("dispatch_stalled"))
    drain_overdue = (
        bool(state.get("drain_active"))
        and float(state.get("drain_active_age_s") or 0.0) >= 90.0
    )
    if (not drain_overdue and not dispatch_stalled) or bool(
        getattr(self, "_entry_eval_stall_disarmed", False)
    ):
        return False
    with self._eval_gate_lock:
        self._entry_eval_stall_disarmed = True
    self._runtime_live_orders_armed = False
    # An evaluator that never receives work is a different fault from a
    # worker stuck mid-drain; name it so operators can tell them apart.
    self._runtime_readiness_reason = (
        "strategy_evaluation_stalled"
        if dispatch_stalled and not drain_overdue
        else "entry_eval_worker_stalled"
    )
    stall_event = (
        "ENTRY_EVAL_DISPATCH_STALLED"
        if dispatch_stalled and not drain_overdue
        else "ENTRY_EVAL_WORKER_STALLED_ENTRIES_DISARMED"
    )
    self._logger.critical(
        "%s symbol=%s phase=%s age_s=%s reason=%s selected_ce=%s selected_pe=%s "
        "tick_age_s=%s last_progress_age_s=%s pending=%s drain_scheduled=%s "
        "drain_active=%s loop_attached=%s",
        stall_event,
        state.get("active_symbol"),
        state.get("active_phase"),
        state.get("drain_active_age_s"),
        self._runtime_readiness_reason,
        getattr(self, "_active_selected_ce", None),
        getattr(self, "_active_selected_pe", None),
        state.get("tick_age_s"),
        state.get("last_progress_age_s"),
        state.get("pending_entry_eval"),
        state.get("drain_scheduled"),
        state.get("drain_active"),
        state.get("runtime_loop_attached"),
        extra={
            "event": stall_event,
            "reason": self._runtime_readiness_reason,
            "selected_ce": getattr(self, "_active_selected_ce", None),
            "selected_pe": getattr(self, "_active_selected_pe", None),
            "last_selected_tick_age_s": (
                round(
                    max(
                        0.0,
                        time.monotonic()
                        - float(
                            getattr(self, "_last_selected_option_tick_ts", 0.0) or 0.0
                        ),
                    ),
                    1,
                )
                if float(getattr(self, "_last_selected_option_tick_ts", 0.0) or 0.0)
                > 0.0
                else None
            ),
            "last_datahub_runner_delivery_at": getattr(
                getattr(self, "_data_hub", None),
                "_last_datahub_runner_delivery_at",
                None,
            ),
            "runner_tick_received_count": getattr(
                self, "_runner_tick_received_count", 0
            ),
            "entry_eval_enqueue_count": getattr(self, "_entry_eval_enqueue_count", 0),
            "entry_eval_started_count": getattr(self, "_entry_eval_started_count", 0),
            "entry_eval_completed_count": getattr(
                self, "_entry_eval_completed_count", 0
            ),
            "selected_candidate_eval_completed_count": getattr(
                self, "_selected_candidate_eval_completed_count", 0
            ),
            "build_sha": (getattr(self, "_build_info", {}) or {}).get("git_sha"),
            **dict(state),
        },
    )
    return True


def get_entry_eval_lifecycle_snapshot(self: Any) -> dict[str, Any]:
    """Cheap counters/timestamps for /health/trading and diagnostics."""
    now = time.monotonic()
    liveness = self._entry_eval_liveness_snapshot(now)
    hub = getattr(self, "_data_hub", None)
    mdm = getattr(self, "_market_data", None)
    selected_tick_at = float(getattr(self, "_last_selected_option_tick_ts", 0.0) or 0.0)
    selected_eval_at = float(
        getattr(self, "_last_selected_candidate_eval_completed_ts", 0.0) or 0.0
    )
    last_eval_at = float(getattr(self, "_last_entry_eval_completed_at", 0.0) or 0.0)
    stall_s = float(getattr(self, "_entry_eval_dispatch_stall_s", 120.0) or 120.0)
    evaluation_alive = bool(liveness.get("evaluation_alive"))
    return {
        "evaluation_ready": bool(getattr(self, "_runtime_evaluation_ready", False)),
        "evaluation_alive": evaluation_alive,
        "evaluation_age_s": (
            round(max(0.0, now - last_eval_at), 1) if last_eval_at > 0.0 else None
        ),
        "selected_eval_age_s": liveness.get("selected_eval_age_s"),
        "strategy_evaluation_stalled": bool(liveness.get("dispatch_stalled")),
        "entry_eval_worker_stalled": bool(liveness.get("worker_stalled")),
        "mdm_selected_tick_count": int(
            getattr(mdm, "_mdm_selected_tick_count", 0) or 0
        ),
        "last_mdm_selected_tick_at": getattr(mdm, "_last_mdm_selected_tick_at", None),
        "datahub_runner_delivery_count": int(
            getattr(hub, "_datahub_runner_delivery_count", 0) or 0
        ),
        "last_datahub_runner_delivery_at": getattr(
            hub, "_last_datahub_runner_delivery_at", None
        ),
        "runner_tick_received_count": int(
            getattr(self, "_runner_tick_received_count", 0) or 0
        ),
        "last_runner_tick_received_at": getattr(
            self, "_last_runner_tick_received_at", None
        ),
        "entry_eligible_tick_count": int(
            getattr(self, "_entry_eligible_tick_count", 0) or 0
        ),
        "last_entry_eligible_tick_at": getattr(
            self, "_last_entry_eligible_tick_at", None
        ),
        "entry_eval_enqueue_count": int(
            getattr(self, "_entry_eval_enqueue_count", 0) or 0
        ),
        "last_entry_eval_enqueued_at": getattr(
            self, "_last_entry_eval_enqueued_at", None
        ),
        "entry_eval_started_count": int(
            getattr(self, "_entry_eval_started_count", 0) or 0
        ),
        "last_entry_eval_started_at": getattr(
            self, "_last_entry_eval_started_at", None
        ),
        "entry_eval_completed_count": int(
            getattr(self, "_entry_eval_completed_count", 0) or 0
        ),
        "last_entry_eval_completed_at": getattr(
            self, "_last_entry_eval_completed_at", None
        ),
        "selected_candidate_eval_completed_count": int(
            getattr(self, "_selected_candidate_eval_completed_count", 0) or 0
        ),
        "last_selected_candidate_eval_completed_at": selected_eval_at or None,
        "last_selected_option_tick_age_s": (
            round(max(0.0, now - selected_tick_at), 1)
            if selected_tick_at > 0.0
            else None
        ),
        "dispatch_stall_s": stall_s,
        **liveness,
    }
