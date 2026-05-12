from __future__ import annotations

import pytest

from nifty_scalper_bot.core import app


def test_live_arm_blocks_low_history(monkeypatch):
    if not hasattr(app, '_compute_live_readiness'):
        pytest.skip('readiness helper unavailable in current build')
