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


async def test_status_json_uses_no_subprocess(monkeypatch) -> None:
    # The Logs page polls status.json every few seconds. It must derive health
    # from the in-memory log tail (the dashboard runs inside the bot process, so
    # answering at all means it's alive) and spawn NO subprocess — the old
    # systemctl-per-poll saturated the threadpool and hung the dashboard.
    import json as _json

    import nifty_scalper_bot.admin_dashboard as dash

    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise AssertionError("status.json must not spawn a subprocess")

    monkeypatch.setattr(dash.subprocess, "run", _boom)
    monkeypatch.setattr(dash.subprocess, "Popen", _boom)
    monkeypatch.setattr(dash, "_gather_logs", lambda *_a, **_k: "fully operational")
    monkeypatch.setattr(dash, "_check_auth", lambda _r: None)
    dash._STATUS_CACHE.update({"at": 0.0})  # force a cold call

    resp = dash.status_json(request=None)
    assert calls["n"] == 0
    assert _json.loads(bytes(resp.body))["label"] == "operational"


async def test_read_env_is_cached(tmp_path, monkeypatch) -> None:
    # Env must not be re-read from disk on every request (disk is slow under swap
    # pressure; repeated reads contributed to the hang). Second call within TTL
    # returns the cached copy without touching disk.
    import nifty_scalper_bot.admin_dashboard as dash

    envf = tmp_path / ".env"
    envf.write_text("KITE_API_KEY=secret\n")
    monkeypatch.setattr(dash, "ENV_PATH", envf)
    dash._ENV_CACHE.update({"at": 0.0, "data": {}})

    reads = {"n": 0}
    orig_read_text = type(envf).read_text
    def _counting_read_text(self, *a, **k):
        reads["n"] += 1
        return orig_read_text(self, *a, **k)
    monkeypatch.setattr(type(envf), "read_text", _counting_read_text)

    assert dash._read_env()["KITE_API_KEY"] == "secret"
    dash._read_env(); dash._read_env()
    assert reads["n"] == 1, "env should be read from disk once, then cached"


async def test_no_session_machinery_remains() -> None:
    # Password / sessions were removed; the helpers must be gone.
    import nifty_scalper_bot.admin_dashboard as dash
    assert not hasattr(dash, "_session_add")
    assert not hasattr(dash, "_SESSIONS")


async def test_trades_json_filters_trade_events(tmp_path, monkeypatch) -> None:
    # The trades endpoint returns only trade-relevant lines so the trade can be
    # reviewed without scrolling the whole log.
    import nifty_scalper_bot.admin_dashboard as dash
    sample = "\n".join([
        "[2026-06-18 13:20:08 IST] ORDER_SENT symbol=NFO:NIFTY26JUN24150PE side=BUY qty=65",
        "[2026-06-18 13:20:09 IST] RUNNER_NO_TRADE_DECISION symbol=X reason=no_vote",
        "[2026-06-18 13:25:00 IST] EXIT symbol=NFO:NIFTY26JUN24150PE pnl=325.0",
    ])
    monkeypatch.setattr(dash, "_gather_logs", lambda *a, **k: sample)
    monkeypatch.setattr(dash, "_check_auth", lambda _r: None)
    out = dash.trades_json(request=None).body.decode()
    assert "ORDER_SENT" in out and "EXIT" in out
    assert "RUNNER_NO_TRADE_DECISION" not in out  # noise filtered out


async def test_app_uses_normalize_symbol_not_datahub_normalize() -> None:
    # Regression: position sync/adoption called DataHub.normalize (does not exist),
    # raising 'type object DataHub has no attribute normalize' and aborting adoption.
    # Must use the normalize_symbol helper instead.
    import pathlib
    src = pathlib.Path("src/nifty_scalper_bot/core/app.py").read_text()
    assert "DataHub.normalize(" not in src
    assert "norm_symbol = normalize_symbol(raw_symbol)" in src
    from nifty_scalper_bot.utils.symbols import normalize_symbol
    assert normalize_symbol("nfo:nifty2662324100ce") == "NFO:NIFTY2662324100CE"


async def test_gather_logs_uses_oneoff_subprocess(monkeypatch) -> None:
    # Super-lite: no background follower, no ring. _gather_logs does a single
    # one-off journalctl read on demand (download-on-click). No background thread
    # runs during market hours, so the admin process can't add load.
    import nifty_scalper_bot.admin_dashboard as dash

    assert not hasattr(dash, "_LOG_RING")
    assert not hasattr(dash, "_log_follower_loop")

    class _O:
        returncode = 0
        stdout = "[2026-06-18 14:00:00 IST] ORDER_SENT x\n"

    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        return _O()

    monkeypatch.setattr(dash.subprocess, "run", _run)
    out = dash._gather_logs(400)
    assert calls["n"] == 1
    assert "ORDER_SENT x" in out

