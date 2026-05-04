from __future__ import annotations

from pathlib import Path


def test_deployment_files_do_not_use_entrypoint() -> None:
    dockerfile = Path('Dockerfile').read_text(encoding='utf-8')
    railway = Path('railway.toml').read_text(encoding='utf-8')

    assert 'entrypoint.sh' not in dockerfile
    assert '/bin/bash /app/entrypoint.sh' not in railway
    assert '[env]' not in railway
    assert 'source = ".env"' not in railway
