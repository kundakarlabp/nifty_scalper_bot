from __future__ import annotations

from pathlib import Path


manager_path = Path("src/nifty_scalper_bot/core/strategy_manager.py")
manager = manager_path.read_text(encoding="utf-8")
start_marker = '        metadata["final_trade_score"] = round(final_score, 3)\n'
end_marker = '        quality_score, quality_meta = self._compute_trade_quality_score(\n'

if 'metadata["manager_final_score_reference_only"] = True' not in manager:
    start = manager.index(start_marker)
    end = manager.index(end_marker, start)
    replacement = '''        metadata["final_trade_score"] = round(final_score, 3)
        metadata["final_trade_threshold_reference"] = round(threshold, 3)
        metadata["manager_final_score_reference_only"] = True
        metadata["manager_final_score_reference_pass"] = final_score >= threshold
        if vetoed:
            blocked_reason = "hard_context_veto"
            log_throttled_live(
                log,
                logging.INFO,
                "TRADE_DECISION_TRACE",
                f"TRADE_DECISION_TRACE:{best_vote.strategy}:{symbol_norm}:{blocked_reason}",
                float(
                    os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120")
                    or "120"
                ),
                "TRADE_DECISION_TRACE symbol=%s strategy=%s side=%s "
                "data_gate=%s final_score=%.2f reference_min=%.2f allowed=%s "
                "blocked_at=%s blocked_reason=%s",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                True,
                final_score,
                threshold,
                False,
                "strategy_manager_combine",
                blocked_reason,
                extra={
                    "event": "TRADE_DECISION_TRACE",
                    "symbol": symbol_norm,
                    "strategy": best_vote.strategy,
                    "side": best_vote.side,
                    "final_score": final_score,
                    "reference_min": threshold,
                    "allowed": False,
                    "blocked_at": "strategy_manager_combine",
                    "blocked_reason": blocked_reason,
                },
            )
            record_strategy_evaluation(
                strategy=str(best_vote.strategy),
                symbol=symbol_norm,
                accepted=False,
                reason=blocked_reason,
                score=final_score,
            )
            maybe_emit_strategy_rejection_summary(log, interval_seconds=300.0)
            _record_no_signal(
                "strategy_score_below_threshold",
                blocked_reason,
                "strategy_manager_combine",
                trigger_vote_count=len(trigger_votes),
                context_vote_count=len(context_votes),
            )
            return None
        if final_score < threshold:
            log_throttled_live(
                log,
                logging.INFO,
                "STRATEGY_MANAGER_SCORE_REFERENCE_BELOW_MIN",
                f"STRATEGY_MANAGER_SCORE_REFERENCE_BELOW_MIN:{best_vote.strategy}:{symbol_norm}",
                float(
                    os.getenv("LOG_THROTTLE_STRATEGY_REJECT_SECONDS", "120")
                    or "120"
                ),
                "STRATEGY_MANAGER_SCORE_REFERENCE_BELOW_MIN symbol=%s "
                "strategy=%s side=%s score=%.2f reference_min=%.2f "
                "final_owner=runner",
                symbol_norm,
                best_vote.strategy,
                best_vote.side,
                final_score,
                threshold,
                extra={
                    "event": "STRATEGY_MANAGER_SCORE_REFERENCE_BELOW_MIN",
                    "symbol": symbol_norm,
                    "strategy": best_vote.strategy,
                    "side": best_vote.side,
                    "score": final_score,
                    "reference_min": threshold,
                    "final_owner": "runner_final_execution_score",
                },
            )
'''
    manager = manager[:start] + replacement + manager[end:]
    manager_path.write_text(manager, encoding="utf-8")

assert "if vetoed or final_score < threshold:" not in manager_path.read_text(encoding="utf-8")
assert 'metadata["manager_final_score_reference_only"] = True' in manager_path.read_text(encoding="utf-8")


test_path = Path("tests/core/test_postmerge_trade_path.py")
test_text = test_path.read_text(encoding="utf-8")
test_name = "test_manager_final_trade_score_is_reference_only_runner_owns_numeric_quality"
if test_name not in test_text:
    test_text += '''\n\ndef test_manager_final_trade_score_is_reference_only_runner_owns_numeric_quality(monkeypatch) -> None:\n    monkeypatch.setenv("EXECUTION_MODE", "LIVE")\n    monkeypatch.setenv("ENABLE_LIVE", "true")\n    monkeypatch.setenv("STRATEGY_TRIGGER_MIN_SCORE", "4.5")\n    manager = StrategyManager.__new__(StrategyManager)\n    manager._last_no_signal_decision_by_symbol = {}\n\n    vwap = _signal_vote(\n        "VWAPPro", raw_score=8.5, weighted_score=5.5, confidence=0.90\n    )\n    orb = _signal_vote(\n        "ORBPro", raw_score=8.0, weighted_score=5.0, confidence=0.85\n    )\n    opposing_context = _signal_vote(\n        "OrderFlow",\n        side="PE",\n        raw_score=3.0,\n        weighted_score=3.0,\n        confidence=0.70,\n        role="context",\n    )\n    opposing_context[1].metadata["context_veto_score"] = 3.0\n\n    result = manager._combine_strategy_votes(\n        symbol=_SYMBOL,\n        signals=[vwap, orb, opposing_context],\n        indicators=_live_indicators(),\n    )\n\n    assert result is not None\n    assert result.metadata["final_trade_score"] < 4.5\n    assert result.metadata["manager_final_score_reference_only"] is True\n    assert result.metadata["manager_final_score_reference_pass"] is False\n    assert result.metadata["quality_gate_owner"] == "runner_final_execution_score"\n'''
    test_path.write_text(test_text, encoding="utf-8")

print("manager final score migration prepared")
