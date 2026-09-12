from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    assert count == 1, f"{label}: expected exactly one match, found {count}"
    return text.replace(old, new, 1)


# StrategyManager: native policy ownership + diagnostic-only manager quality score.
manager_path = Path("src/nifty_scalper_bot/core/strategy_manager.py")
manager = manager_path.read_text(encoding="utf-8")

manager = replace_once(
    manager,
    "from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager\n"
    "from nifty_scalper_bot.core.underlying_direction import (\n",
    "from nifty_scalper_bot.core.market_regime_manager import MarketRegimeManager\n"
    "from nifty_scalper_bot.core.strategy_vote_policy import (\n"
    "    independent_same_side_confirmation,\n"
    "    is_permanent_context_only,\n"
    "    partition_votes,\n"
    ")\n"
    "from nifty_scalper_bot.core.underlying_direction import (\n",
    "strategy-manager policy imports",
)

old_partition = '''        mode_profile = self.get_strategy_mode_profile()
        trigger_votes: list[tuple[Signal, StrategyVote]] = []
        context_votes: list[tuple[Signal, StrategyVote]] = []
        for signal, vote in signals:
            if signal.action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                signal.metadata = dict(signal.metadata or {})
                signal.metadata["approval_path"] = "close_signal"
                log.info("TRADE_DECISION_TRACE approval_path=%s symbol=%s", "close_signal", symbol_norm)
                return signal
            role = str((vote.metadata or {}).get("role") or "trigger").lower()
            if role == "context":
                context_votes.append((signal, vote))
            else:
                trigger_votes.append((signal, vote))
'''
new_partition = '''        mode_profile = self.get_strategy_mode_profile()
        entry_signals: list[tuple[Signal, StrategyVote]] = []
        for signal, vote in signals:
            if signal.action in {"CLOSE_LONG", "CLOSE_SHORT"}:
                signal.metadata = dict(signal.metadata or {})
                signal.metadata["approval_path"] = "close_signal"
                log.info("TRADE_DECISION_TRACE approval_path=%s symbol=%s", "close_signal", symbol_norm)
                return signal
            entry_signals.append((signal, vote))

        trigger_votes, context_votes, rejected_setups = partition_votes(entry_signals)
        if rejected_setups and not trigger_votes:
            log_throttled(
                log,
                f"strategy_setup_contract_failed:{symbol_norm}",
                "STRATEGY_SETUP_CONTRACT_FAILED symbol=%s rejected=%s",
                symbol_norm,
                rejected_setups,
                interval_sec=30.0,
                level=logging.INFO,
                extra={
                    "event": "STRATEGY_SETUP_CONTRACT_FAILED",
                    "symbol": symbol_norm,
                    "rejected": rejected_setups,
                },
            )
            _record_no_signal(
                "strategy_setup_rejected",
                "setup_contract_failed",
                "strategy_setup_gate",
                trigger_vote_count=0,
                context_vote_count=len(context_votes),
                final_block_reason="setup_contract_failed",
            )
            return None

        trigger_sides = {
            str(vote.side or "").upper()
            for _signal, vote in trigger_votes
            if str(vote.side or "").upper() in {"CE", "PE"}
        }
        if len(trigger_sides) > 1:
            _record_no_signal(
                "strategy_conflict",
                "conflicting_trigger_direction",
                "trigger_direction_gate",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
                final_block_reason="conflicting_trigger_direction",
            )
            log_throttled(
                log,
                f"conflicting_trigger_direction:{symbol_norm}",
                "CONFLICTING_TRIGGER_DIRECTION_BLOCKED symbol=%s sides=%s strategies=%s",
                symbol_norm,
                sorted(trigger_sides),
                [vote.strategy for _signal, vote in trigger_votes],
                interval_sec=30.0,
                level=logging.INFO,
                extra={
                    "event": "CONFLICTING_TRIGGER_DIRECTION_BLOCKED",
                    "symbol": symbol_norm,
                    "sides": sorted(trigger_sides),
                    "strategies": [vote.strategy for _signal, vote in trigger_votes],
                },
            )
            return None

        confirmation, confirmation_strategies = independent_same_side_confirmation(
            entry_signals
        )
        if confirmation:
            indicator_map["independent_trigger_confirmation"] = True
            indicator_map["independent_trigger_confirmation_strategies"] = (
                confirmation_strategies
            )
'''
manager = replace_once(manager, old_partition, new_partition, "native vote partition")

manager = replace_once(
    manager,
    '''        if not context_votes:
            return None
        if not bool(mode_profile.get("allow_context_promotion", True)):
''',
    '''        if not context_votes:
            return None
        context_votes = [
            pair for pair in context_votes if not is_permanent_context_only(pair[1])
        ]
        if not context_votes:
            return None
        if not bool(mode_profile.get("allow_context_promotion", True)):
''',
    "native permanent-context promotion block",
)

quality_start = manager.index("    def _compute_trade_quality_score(")
quality_end = manager.index("\n    def increment_observability_counter", quality_start)
quality_block = manager[quality_start:quality_end]
quality_block = replace_once(
    quality_block,
    "        score = max(0.0, min(10.0, sum(components.values()) + sum(penalties.values())))\n",
    '''        confirmation_bonus = (
            0.5
            if bool(indicators.get("independent_trigger_confirmation"))
            and not already_blocked_by_strategy
            else 0.0
        )
        components["independent_trigger_confirmation"] = confirmation_bonus
        score = max(0.0, min(10.0, sum(components.values()) + sum(penalties.values())))
''',
    "native confirmation quality evidence",
)
quality_block = replace_once(
    quality_block,
    '            "strategy_block_reason": strategy_block_reason or None,\n'
    '            "trade_quality_symbol": symbol,\n',
    '''            "strategy_block_reason": strategy_block_reason or None,
            "independent_trigger_confirmation": bool(confirmation_bonus),
            "independent_trigger_confirmation_strategies": list(
                indicators.get("independent_trigger_confirmation_strategies") or []
            ),
            "trade_quality_symbol": symbol,
''',
    "native confirmation quality metadata",
)
manager = manager[:quality_start] + quality_block + manager[quality_end:]

q_start_marker = '        quality_min_required = float(mode_profile.get("min_trade_quality", 5.0))\n'
q_end_marker = '        direction_bias = str(indicator_map.get("direction_bias") or indicator_map.get("underlying_direction_bias") or "").upper()\n'
q_start = manager.index(q_start_marker)
q_end = manager.index(q_end_marker, q_start)
manager_quality_block = '''        quality_min_required = float(mode_profile.get("min_trade_quality", 5.0))
        quality_pass = quality_score >= quality_min_required
        metadata.update(quality_meta)
        metadata["quality_min_required"] = quality_min_required
        metadata["quality_pass"] = quality_pass
        metadata["quality_gate_owner"] = "runner_final_execution_score"
        metadata["manager_quality_reference_only"] = True

        explicit_strategy_block = bool(quality_meta.get("already_blocked_by_strategy"))
        if explicit_strategy_block:
            blocked_reason = str(
                quality_meta.get("strategy_block_reason")
                or quality_meta.get("quality_block_reason")
                or "strategy_evidence_invalid"
            )
            log_throttled_live(
                log,
                logging.INFO,
                "STRATEGY_EXPLICIT_BLOCK",
                f"STRATEGY_EXPLICIT_BLOCK:{best_vote.strategy}:{symbol_norm}:{blocked_reason}",
                float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
                "STRATEGY_EXPLICIT_BLOCK symbol=%s strategy=%s side=%s reason=%s",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                blocked_reason,
                extra={
                    "event": "STRATEGY_EXPLICIT_BLOCK",
                    "symbol": symbol_norm,
                    "strategy": best_vote.strategy,
                    "side": best_vote.side,
                    "reason": blocked_reason,
                },
            )
            record_strategy_evaluation(
                strategy=str(best_vote.strategy),
                symbol=symbol_norm,
                accepted=False,
                reason=blocked_reason,
                score=quality_score,
            )
            maybe_emit_strategy_rejection_summary(log, interval_seconds=300.0)
            _record_no_signal(
                "strategy_no_trigger",
                blocked_reason,
                "strategy_explicit_block",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
                final_block_reason=blocked_reason,
            )
            return None

        if not quality_pass:
            log_throttled_live(
                log,
                logging.INFO,
                "STRATEGY_QUALITY_REFERENCE_BELOW_MIN",
                f"STRATEGY_QUALITY_REFERENCE_BELOW_MIN:{best_vote.strategy}:{symbol_norm}",
                float(os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120") or "120"),
                "STRATEGY_QUALITY_REFERENCE_BELOW_MIN symbol=%s strategy=%s side=%s score=%.2f reference_min=%.2f final_owner=runner",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                quality_score,
                quality_min_required,
                extra={
                    "event": "STRATEGY_QUALITY_REFERENCE_BELOW_MIN",
                    "symbol": symbol_norm,
                    "strategy": best_vote.strategy,
                    "side": best_vote.side,
                    "score": quality_score,
                    "reference_min": quality_min_required,
                    "final_owner": "runner_final_execution_score",
                },
            )

'''
manager = manager[:q_start] + manager_quality_block + manager[q_end:]
manager = replace_once(
    manager,
    "            and quality_pass\n            and not hard_veto_reasons\n",
    "            and not hard_veto_reasons\n",
    "remove diagnostic manager quality from consensus approval",
)
manager_path.write_text(manager, encoding="utf-8")


# Canonical vote policy: expose permanent role as a pure helper.
policy_path = Path("src/nifty_scalper_bot/core/strategy_vote_policy.py")
policy = policy_path.read_text(encoding="utf-8")
policy = replace_once(
    policy,
    '''def vote_role(vote: Any) -> str:
    """Return the effective immutable role for a vote."""
    strategy = str(getattr(vote, "strategy", "") or "").strip().lower()
    if strategy in _CONTEXT_ONLY_STRATEGIES:
        return "context"
''',
    '''def is_permanent_context_only(vote: Any) -> bool:
    """Return whether a strategy is structurally context-only."""
    strategy = str(getattr(vote, "strategy", "") or "").strip().lower()
    return strategy in _CONTEXT_ONLY_STRATEGIES


def vote_role(vote: Any) -> str:
    """Return the effective immutable role for a vote."""
    if is_permanent_context_only(vote):
        return "context"
''',
    "pure permanent context role helper",
)
policy = replace_once(
    policy,
    '    "independent_same_side_confirmation",\n    "is_close_signal",\n',
    '    "independent_same_side_confirmation",\n    "is_close_signal",\n    "is_permanent_context_only",\n',
    "policy exports",
)
policy_path.write_text(policy, encoding="utf-8")


# Compatibility helpers remain importable, but no longer mutate StrategyManager.
setup_path = Path("src/nifty_scalper_bot/core/strategy_setup_score_gate.py")
setup_path.write_text(
    '''"""Compatibility helpers for the canonical strategy-vote policy.

Runtime ownership lives directly in :mod:`strategy_manager`; this module contains
no import-time installers or class mutation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from nifty_scalper_bot.core.strategy_vote_policy import (
    is_permanent_context_only,
    setup_gate_decision,
    vote_role,
)


def setup_gate_result(vote: Any) -> tuple[bool, float | None, float | None, str | None]:
    decision = setup_gate_decision(vote)
    return decision.passed, decision.score, decision.minimum, decision.reason


def enforce_context_only_role(signal: Any, vote: Any) -> tuple[Any, bool]:
    """Normalize malformed permanent context-only metadata without mutation hooks."""
    if not is_permanent_context_only(vote) or getattr(signal, "action", None) in {
        "CLOSE_LONG",
        "CLOSE_SHORT",
    }:
        return signal, False
    enforced = {
        "role": "context",
        "can_trigger": False,
        "trigger_conditions_met": False,
        "trigger_eligible": False,
        "trigger_block_reason": "context_only_role",
        "trigger_disqualified_by": "context_only_role",
        "context_role": "confirmation",
    }
    vote_metadata = dict(getattr(vote, "metadata", {}) or {})
    signal_metadata = dict(getattr(signal, "metadata", {}) or {})
    changed = any(
        vote_metadata.get(key) != value or signal_metadata.get(key) != value
        for key, value in enforced.items()
    )
    vote_metadata.update(enforced)
    signal_metadata.update(enforced)
    vote.metadata = vote_metadata
    with_metadata = getattr(signal, "with_metadata", None)
    if callable(with_metadata):
        signal = with_metadata(**enforced)
    elif dataclasses.is_dataclass(signal):
        signal = dataclasses.replace(signal, metadata=signal_metadata)
    else:
        signal.metadata = signal_metadata
    return signal, changed


def filter_context_promotions(
    context_votes: list[tuple[Any, Any]],
) -> tuple[list[tuple[Any, Any]], list[str]]:
    """Return promotable context votes plus permanent-context exclusions."""
    eligible: list[tuple[Any, Any]] = []
    blocked: list[str] = []
    for signal, vote in context_votes:
        if is_permanent_context_only(vote):
            blocked.append(str(getattr(vote, "strategy", "") or "unknown"))
        else:
            eligible.append((signal, vote))
    return eligible, blocked


__all__ = [
    "enforce_context_only_role",
    "filter_context_promotions",
    "setup_gate_result",
    "vote_role",
]
''',
    encoding="utf-8",
)


# Remove the import-time setup gate installer from core package initialization.
init_path = Path("src/nifty_scalper_bot/core/__init__.py")
init_text = init_path.read_text(encoding="utf-8")
patch_start = init_text.index(
    "try:\n    from nifty_scalper_bot.core.strategy_setup_score_gate import ("
)
patch_end = init_text.index(
    "\ntry:\n    from nifty_scalper_bot.core.boot_log_safety import (", patch_start
)
init_text = init_text[:patch_start] + init_text[patch_end + 1 :]
assert "_apply_strategy_setup_score_gate" not in init_text
init_path.write_text(init_text, encoding="utf-8")


# Remove two trading-path monkey patches whose behavior now lives natively.
reliability_path = Path("src/nifty_scalper_bot/core/runtime_reliability_hardening.py")
reliability = reliability_path.read_text(encoding="utf-8")
patch_start = reliability.index("def _trigger_confirmation_details(")
patch_end = reliability.index("def _is_dynamic_option_symbol(", patch_start)
reliability = reliability[:patch_start] + reliability[patch_end:]
reliability = reliability.replace("from dataclasses import replace\n", "", 1)
for line in (
    '        "trade_quality": _install_trade_quality_patch(),\n',
    '        "strategy_reason": _install_strategy_reason_patch(),\n',
    '    "_trigger_confirmation_details",\n',
):
    assert line in reliability, f"missing reliability line: {line!r}"
    reliability = reliability.replace(line, "", 1)
assert "_install_trade_quality_patch" not in reliability
assert "_install_strategy_reason_patch" not in reliability
reliability_path.write_text(reliability, encoding="utf-8")


# Focused regression expectations.
postmerge_path = Path("tests/core/test_postmerge_trade_path.py")
postmerge = postmerge_path.read_text(encoding="utf-8")
postmerge = replace_once(
    postmerge,
    '''    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_SYMBOL]
    assert decision.blocked_at == "trade_quality_gate"
''',
    '''    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_SYMBOL]
    assert decision.blocked_at == "trigger_direction_gate"
    assert decision.reason == "conflicting_trigger_direction"
''',
    "opposite trigger explicit conflict regression",
)
postmerge += '''

def test_manager_quality_reference_is_diagnostic_runner_owns_final_score(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}
    manager._compute_trade_quality_score = lambda *args, **kwargs: (
        3.0,
        {
            "trade_quality_score": 3.0,
            "trade_quality_components": {},
            "trade_quality_penalties": {},
            "quality_block_reason": "ok",
            "already_blocked_by_strategy": False,
            "strategy_block_reason": None,
        },
    )
    vwap = _signal_vote(
        "VWAPPro", raw_score=8.5, weighted_score=8.5, confidence=0.90
    )
    orb = _signal_vote(
        "ORBPro", raw_score=8.0, weighted_score=8.0, confidence=0.85
    )

    result = manager._combine_strategy_votes(
        symbol=_SYMBOL,
        signals=[vwap, orb],
        indicators=_live_indicators(),
    )

    assert result is not None
    assert result.metadata["quality_pass"] is False
    assert result.metadata["manager_quality_reference_only"] is True
    assert result.metadata["quality_gate_owner"] == "runner_final_execution_score"


def test_structural_strategy_invalid_state_remains_a_hard_block(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ENABLE_LIVE", "true")
    manager = StrategyManager.__new__(StrategyManager)
    manager._last_no_signal_decision_by_symbol = {}
    manager._compute_trade_quality_score = lambda *args, **kwargs: (
        9.0,
        {
            "trade_quality_score": 9.0,
            "trade_quality_components": {},
            "trade_quality_penalties": {},
            "quality_block_reason": "quote_depth_invalid",
            "already_blocked_by_strategy": True,
            "strategy_block_reason": None,
        },
    )
    vwap = _signal_vote(
        "VWAPPro", raw_score=8.5, weighted_score=8.5, confidence=0.90
    )
    orb = _signal_vote(
        "ORBPro", raw_score=8.0, weighted_score=8.0, confidence=0.85
    )

    result = manager._combine_strategy_votes(
        symbol=_SYMBOL,
        signals=[vwap, orb],
        indicators=_live_indicators(),
    )

    assert result is None
    decision = manager._last_no_signal_decision_by_symbol[_SYMBOL]
    assert decision.blocked_at == "strategy_explicit_block"
'''
postmerge_path.write_text(postmerge, encoding="utf-8")

setup_test_path = Path("tests/core/test_strategy_setup_score_gate.py")
setup_test = setup_test_path.read_text(encoding="utf-8")
setup_test += '''

def test_setup_policy_is_native_not_import_time_patched() -> None:
    from pathlib import Path

    core_init = Path("src/nifty_scalper_bot/core/__init__.py").read_text(encoding="utf-8")
    setup_module = Path(
        "src/nifty_scalper_bot/core/strategy_setup_score_gate.py"
    ).read_text(encoding="utf-8")
    reliability = Path(
        "src/nifty_scalper_bot/core/runtime_reliability_hardening.py"
    ).read_text(encoding="utf-8")

    assert "_apply_strategy_setup_score_gate" not in core_init
    assert "def apply_patches" not in setup_module
    assert "_install_trade_quality_patch" not in reliability
    assert "_install_strategy_reason_patch" not in reliability
'''
setup_test_path.write_text(setup_test, encoding="utf-8")

print("entry admission migration prepared")
