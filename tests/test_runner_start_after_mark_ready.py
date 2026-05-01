from pathlib import Path


def test_mark_ready_precedes_startup_runner_start_reason() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text()
    assert text.index('runner.mark_ready(readiness_symbols)') < text.index('reason="startup_mark_ready_complete"')
