from __future__ import annotations

import logging

from nifty_scalper_bot.utils.logging import setup_logging


def test_info_goes_stdout_and_error_goes_stderr(capsys, monkeypatch) -> None:
    monkeypatch.setenv("LOG_DEDUP_ENABLED", "0")
    setup_logging("INFO")
    logger = logging.getLogger("nifty_scalper_bot.test_logging_severity")

    logger.info("info-severity-check")
    logger.error("error-severity-check")

    captured = capsys.readouterr()
    assert "info-severity-check" in captured.out
    assert "info-severity-check" not in captured.err
    assert "error-severity-check" in captured.err
