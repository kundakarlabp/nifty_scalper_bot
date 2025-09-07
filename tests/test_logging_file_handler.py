from __future__ import annotations

import logging
from importlib import reload

import src.main as main


def test_setup_logging_adds_file_handler(tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "bot.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))

    root = logging.getLogger()
    prev = list(root.handlers)
    for h in list(root.handlers):
        root.removeHandler(h)

    reload(main)
    main._setup_logging()
    assert any(isinstance(h, logging.FileHandler) for h in root.handlers)

    for h in list(root.handlers):
        root.removeHandler(h)
    for h in prev:
        root.addHandler(h)
