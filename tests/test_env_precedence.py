from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env_defaults(env_path: Path) -> None:
    """Load .env as defaults only. Args: env_path. Returns: None. Raises: none."""
    load_dotenv(dotenv_path=str(env_path), override=False)


def test_system_env_overrides_dotenv_defaults(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / '.env'
    env_file.write_text('ENABLE_LIVE=false\nEXECUTION_MODE=PAPER\n', encoding='utf-8')

    monkeypatch.setenv('ENABLE_LIVE', 'true')
    monkeypatch.setenv('EXECUTION_MODE', 'LIVE')

    load_env_defaults(env_file)

    assert os.getenv('ENABLE_LIVE') == 'true'
    assert os.getenv('EXECUTION_MODE') == 'LIVE'
