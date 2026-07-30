from __future__ import annotations

from pathlib import Path


def test_runner_order_readiness_uses_canonical_execution_quote_readiness() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )
    function_start = source.index("def _ensure_symbol_execution_ready_result")
    function_end = source.index("def _execution_reject_cooldown_result", function_start)
    body = source[function_start:function_end]

    assert "evaluate_execution_quote(" in body

def test_runner_uses_canonical_order_quote_age_setting_at_every_stage() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(
        encoding="utf-8"
    )

    assert 'os.getenv("ORDER_MAX_QUOTE_AGE_MS"' not in source
    assert source.count("app_settings.ORDER_MAX_QUOTE_AGE_MS") == 4
