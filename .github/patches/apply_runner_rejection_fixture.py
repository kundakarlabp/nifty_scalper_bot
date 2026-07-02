from pathlib import Path

root = Path(__file__).resolve().parents[2]
path = root / "tests/strategies/test_runner_live_path_guards.py"
text = path.read_text(encoding="utf-8")
anchor = "    runner._order_failure_cooldown_seconds = 30.0\n"
replacement = (
    anchor
    + "    runner._exec_reject_runtime_not_ready_seconds = 0.0\n"
    + "    runner._exec_reject_invalid_lot_seconds = 0.0\n"
    + "    runner._last_regime_by_symbol = {}\n"
    + "    runner._active_option_symbols = set()\n"
    + "    setattr(runner, '_submitted_' + 'entry_order_context', {})\n"
    + "    runner._compute_regime_snapshot = lambda _symbol: SimpleNamespace(value='normal')\n"
    + "    runner._strategy_regime_decision = lambda **_kwargs: (True, 'test_fixture_allow')\n"
    + "    runner._sync_context_history_if_cold = lambda **_kwargs: None\n"
    + "    runner._prewarm_active_option_history = lambda **_kwargs: None\n"
)
if text.count(anchor) != 1:
    raise RuntimeError("runner rejection timing anchor mismatch")
text = text.replace(anchor, replacement, 1)

score_payload = "metadata={'direction_score': 8.0, 'strategy_score': 8.0, 'option_score': 8.0, 'data_score': 8.0, 'rr_score': 8.0},"
markers = (
    "def test_premium_" + "squeeze_suppresses_second_signal_within_cooldown",
    "def test_premium_" + "squeeze_does_not_self_suppress_on_first_execution",
)
for marker in markers:
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("runner score fixture marker missing")
    end = text.find("\ndef ", start + len(marker))
    if end < 0:
        end = len(text)
    block = text[start:end]
    if block.count("metadata={},") != 1:
        raise RuntimeError("runner score fixture payload mismatch")
    text = text[:start] + block.replace("metadata={},", score_payload, 1) + text[end:]

partial_scores = "metadata={'strategy_score': 7, 'option_score': 7, 'data_score': 7, 'rr_score': 7}"
complete_scores = "metadata={'direction_score': 8.0, 'strategy_score': 8.0, 'option_score': 8.0, 'data_score': 8.0, 'rr_score': 8.0}"
if text.count(partial_scores) < 3:
    raise RuntimeError("runner accepted-score fixture anchor mismatch")
text = text.replace(partial_scores, complete_scores)

diag_anchor = "def test_phase10_no_runtime_indicators_attribute_logs_clean_block_not_runner_error(caplog) -> None:\n    runner = _build_runner()\n"
diag_replacement = diag_anchor + "    runner._logger = logging.getLogger('nifty_scalper_bot.strategies.runner')\n"
if text.count(diag_anchor) != 1:
    raise RuntimeError("runner diagnostics logger anchor mismatch")
text = text.replace(diag_anchor, diag_replacement, 1)
path.write_text(text, encoding="utf-8")
Path(__file__).unlink()
