from __future__ import annotations

"""Tests for the diagnostics server entry point."""

from unittest.mock import Mock

import pytest

from src.server import main


def test_main_invokes_logging_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    setup_spy = Mock()
    run_spy = Mock()

    monkeypatch.setattr(main, "setup_root_logger", setup_spy)
    monkeypatch.setattr(main, "run_health_server", run_spy)

    main.main()

    setup_spy.assert_called_once_with()
    run_spy.assert_called_once_with()
