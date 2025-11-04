"""
Tests for persistence_state.py: covers save/load, atomicity, error handling,
and backup rotation.
"""
from __future__ import annotations

from pathlib import Path
from nifty_scalper_bot.persistence import persistence_state

# Note: monkeypatch is a pytest fixture injected by pytest

def test_save_and_load_roundtrip(tmp_path: Path):
    file_path = tmp_path / "state.json"
    # Update state
    persistence_state.update_state({
        "orders": {"ORD1": {"status": "open"}},
        "positions": {"POS1": {"qty": 10}},
    })
    persistence_state.dump_state_to_disk(file_path)
    # Clear and reload
    persistence_state.update_state({"orders": {}, "positions": {}})
    persistence_state.load_state_from_disk(file_path)
    with persistence_state.locked_state() as state:
        assert state["orders"] == {"ORD1": {"status": "open"}}
        assert state["positions"] == {"POS1": {"qty": 10}}

def test_atomic_save(tmp_path: Path):
    file_path = tmp_path / "atomic.json"
    persistence_state.update_state({
        "orders": {"ORD2": {"status": "closed"}},
        "positions": {}
    })
    persistence_state.dump_state_to_disk(file_path)
    assert file_path.exists()
    persistence_state.update_state({"orders": {}, "positions": {}})
    persistence_state.load_state_from_disk(file_path)
    with persistence_state.locked_state() as state:
        assert state["orders"]["ORD2"]["status"] == "closed"

def test_backup_rotation(tmp_path: Path):
    file_path = tmp_path / "backup.json"
    persistence_state.update_state({
        "orders": {"ORD3": {"status": "pending"}},
        "positions": {}
    })
    # Save multiple times to trigger backup rotation
    for _ in range(5):
        persistence_state.dump_state_to_disk(file_path)
    backups = list(tmp_path.glob("backup.json.*"))
    assert len(backups) >= 1

def test_load_missing_file(tmp_path: Path):
    file_path = tmp_path / "missing.json"
    # Should not raise, just log and return
    persistence_state.load_state_from_disk(file_path)
    with persistence_state.locked_state() as state:
        assert isinstance(state, dict)

def test_save_error(monkeypatch, tmp_path: Path):
    file_path = tmp_path / "fail.json"
    persistence_state.update_state({"orders": {}, "positions": {}})
    def fail_write(*args: object, **kwargs: object) -> None:
        raise OSError("fail")
    monkeypatch.setattr(Path, "write_bytes", fail_write)
    # Should not raise, just log error
    persistence_state.dump_state_to_disk(file_path)
