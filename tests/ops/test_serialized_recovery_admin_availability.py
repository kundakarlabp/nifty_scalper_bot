"""Verify that operator recovery serializes and admin is engine-independent."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deploy/lightsail_release.sh"
ADMIN = ROOT / "deploy/systemd/niftybot-admin.service"


def _lock_section() -> str:
    source = RELEASE.read_text(encoding="utf-8")
    return source.split('exec 9>"$LOCK_FILE"\n', 1)[1].split('\ncd "$APP_DIR"', 1)[0]


def test_operator_release_waits_for_held_lock(tmp_path: Path) -> None:
    lock = tmp_path / "deploy.lock"
    with lock.open("w") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        snippet = 'exec 9>"$LOCK_FILE"\n' + _lock_section()
        process = subprocess.Popen(
            ["bash", "-c", 'log() { printf "%s\\n" "$*"; }; ' + snippet],
            env={**os.environ, "LOCK_FILE": str(lock), "WAIT_FOR_LOCK": "true"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(0.2)
            assert process.poll() is None, "operator recovery exited instead of waiting"
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)


def test_release_has_bounded_operator_lock_wait() -> None:
    source = RELEASE.read_text(encoding="utf-8")
    assert "--wait-for-lock) WAIT_FOR_LOCK=true" in source
    assert "flock -w 180 9" in source
    assert "exit 75" in source
    assert "flock -n 9" in source
    assert "acquire_release_entry_hold" in source
    assert "intraday_flat_broker_confirmed" in source


def test_admin_is_independent_and_token_form_remains_available() -> None:
    unit = ADMIN.read_text(encoding="utf-8")
    admin = (ROOT / "src/nifty_scalper_bot/admin_dashboard.py").read_text(
        encoding="utf-8"
    )
    operating = (ROOT / "src/nifty_scalper_bot/operating_control.py").read_text(
        encoding="utf-8"
    )
    assert "After=network-online.target\n" in unit
    assert "After=network-online.target niftybot.service" not in unit
    assert "Restart=always" in unit
    assert 'action="/admin/token"' in admin
    assert '@router.post("/admin/token")' in admin
    assert "DEFAULT_START = dt_time(8, 55)" in operating
    assert "DEFAULT_STOP = dt_time(15, 40)" in operating
    assert 'mode == "AUTO" and auto_window_active(now)' in operating
