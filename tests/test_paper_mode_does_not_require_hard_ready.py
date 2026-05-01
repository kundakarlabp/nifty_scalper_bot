from pathlib import Path


def test_paper_gate_starts_runner_when_hard_not_ready() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text()
    assert 'PAPER_EVALUATION_READY hard_ready=%s spot_ready=%s runner_running=%s' in text
    assert 'reason="paper_readiness_recovery"' in text
    assert 'ctx.live_orders_armed = False' in text
