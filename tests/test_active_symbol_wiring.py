from __future__ import annotations

from nifty_scalper_bot.core import app


def test_active_symbol_wiring_helper_exists():
    assert hasattr(app, '_ensure_symbol_pipeline_ready') or hasattr(app, 'ensure_active_symbol_wired')
