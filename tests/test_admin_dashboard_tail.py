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
