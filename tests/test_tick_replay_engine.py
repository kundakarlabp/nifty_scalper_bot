import time
from pathlib import Path

from nifty_scalper_bot.testing.tick_replay_engine import TickReplayEngine


def test_tick_replay_engine_load_and_replay(tmp_path: Path) -> None:
    tick_file = tmp_path / "ticks.csv"
    tick_file.write_text(
        "symbol,ltp,timestamp\nNIFTY,200,1\nNIFTY,201,2\n",
        encoding="utf-8",
    )
    engine = TickReplayEngine(str(tick_file), speed=1_000_000.0)
    engine.load()
    seen: list[dict[str, object]] = []
    engine.replay(seen.append)
    assert len(seen) == 2
    assert seen[0]["ltp"] == 200.0
    assert engine.index == 2


def test_tick_replay_engine_deterministic_mode_skips_sleep(
    tmp_path: Path, monkeypatch
) -> None:
    tick_file = tmp_path / "ticks.csv"
    tick_file.write_text(
        "symbol,ltp,timestamp\nNIFTY,200,1\nNIFTY,201,2\n",
        encoding="utf-8",
    )
    engine = TickReplayEngine(str(tick_file), speed=1.0)
    engine.load()
    calls: list[float] = []

    def _sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(time, "sleep", _sleep)
    seen: list[dict[str, object]] = []
    engine.replay(seen.append, deterministic=True)
    assert len(calls) == 0
    assert len(seen) == 2
