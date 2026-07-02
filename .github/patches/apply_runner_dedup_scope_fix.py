from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "src/nifty_scalper_bot/strategies/runner.py"
text = path.read_text(encoding="utf-8")
anchor = '''            selected_snapshot: dict[str, Any] = {}
            existing_snapshots = (
'''
replacement = '''            selected_snapshot: dict[str, Any] = {}
            dedup_reserved = False
            dedup_key_context: dict[str, str] | None = None

            def reject_after_dedup(
                *,
                reason: str,
                details: Mapping[str, Any] | None = None,
            ) -> SignalExecutionResult:
                if dedup_reserved:
                    self._mark_directional_dedup_failed(
                        underlying=underlying,
                        option_side=option_side,
                        reason=reason_key,
                    )
                self._reset_execution_state(base_symbol)
                return self._reject_signal_execution(
                    symbol=base_symbol,
                    trace_id=trace_id,
                    reason=reason,
                    details=details,
                )

            _reject_after_dedup = reject_after_dedup
            existing_snapshots = (
'''
if text.count(anchor) != 1:
    raise RuntimeError("runner dedup scope anchor mismatch")
path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")

test_path = root / "tests/strategies/test_runner_live_path_guards.py"
test_text = test_path.read_text(encoding="utf-8")
test_anchor = "    runner._active_option_symbols = set()\n"
test_replacement = test_anchor + "    runner._indicator_engine = SimpleNamespace(get_history=lambda _symbol: [])\n"
if test_text.count(test_anchor) != 1:
    raise RuntimeError("runner indicator fixture anchor mismatch")
test_text = test_text.replace(test_anchor, test_replacement, 1)

marker = "def test_candidate_not_in_active_basket_fails_live_entry_eligibility"
start = test_text.find(marker)
end = test_text.find("\ndef ", start + len(marker))
if start < 0 or end < 0:
    raise RuntimeError("inactive basket regression marker missing")
block = test_text[start:end]
index_line = "    runner._active_option_symbols = {\"NFO:NIFTY26JUN23900CE\", \"NFO:NIFTY26JUN23900PE\"}\n"
if block.count(index_line) != 1:
    raise RuntimeError("inactive basket regression index mismatch")
block = block.replace(index_line, index_line + "    runner._active_basket_token_by_symbol = {}\n", 1)
test_text = test_text[:start] + block + test_text[end:]

trigger_old = 'reason="OrderFlow", metadata={"trigger_conditions_met": True})'
trigger_new = 'reason="OrderFlow", stop_loss=100.0, take_profit=120.0, metadata={"trigger_conditions_met": True})'
if test_text.count(trigger_old) != 4:
    raise RuntimeError("trigger-signal fixture anchor mismatch")
test_text = test_text.replace(trigger_old, trigger_new)

test_path.write_text(test_text, encoding="utf-8")
Path(__file__).unlink()
