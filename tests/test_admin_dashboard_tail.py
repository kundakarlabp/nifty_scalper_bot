"""The admin dashboard log viewer must tail the end of bot.log, not load the whole
file into memory on every refresh — that spike was freezing the dashboard on the
memory-tight Lightsail host as the log grew.
"""

from __future__ import annotations

from pathlib import Path

from nifty_scalper_bot.admin_dashboard import _tail_file


async def test_tail_file_returns_last_n_lines(tmp_path: Path) -> None:
    p = tmp_path / "bot.log"
    p.write_text("\n".join(f"line {i}" for i in range(10_000)) + "\n")
    rows = _tail_file(p, 200).splitlines()
    assert len(rows) == 200
    assert rows[-1] == "line 9999"
    assert rows[0] == "line 9800"


async def test_tail_file_is_byte_bounded(tmp_path: Path) -> None:
    # Even asking for many lines, a small byte budget caps how much is read,
    # proving the whole file is never loaded.
    p = tmp_path / "bot.log"
    p.write_text("\n".join(f"line {i}" for i in range(100_000)) + "\n")
    rows = _tail_file(p, 50_000, max_bytes=20_000).splitlines()
    assert 0 < len(rows) < 50_000  # capped by the byte budget, not the line count
    assert rows[-1] == "line 99999"  # still anchored to the file's end


async def test_status_json_caches_subprocess(monkeypatch) -> None:
    # The Logs page polls status.json every few seconds; it must NOT spawn a
    # systemctl subprocess on every poll (that saturated the threadpool and hung
    # the dashboard). Second call within TTL must be served from cache.
    import nifty_scalper_bot.admin_dashboard as dash

    calls = {"n": 0}

    class _Out:
        stdout = "active"

    def _fake_run(*_a, **_k):
        calls["n"] += 1
        return _Out()

    monkeypatch.setattr(dash.subprocess, "run", _fake_run)
    monkeypatch.setattr(dash, "_gather_logs", lambda *_a, **_k: "Bot fully operational")
    monkeypatch.setattr(dash, "_check_auth", lambda _r: None)
    monkeypatch.setenv("ADMIN_STATUS_CACHE_SECONDS", "10")
    dash._STATUS_CACHE.update({"at": 0.0})  # force a cold first call

    dash.status_json(request=None)  # cold -> spawns
    dash.status_json(request=None)  # cached -> no spawn
    dash.status_json(request=None)  # cached -> no spawn
    assert calls["n"] == 1, "status.json must cache, not spawn systemctl every poll"
