from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "nifty_scalper_bot"


MUTATION_PATTERNS = {
    "completed_append": re.compile(r"_completed_candles\s*\.\s*(append|extend)\s*\("),
    "completed_assign": re.compile(r"_completed_candles\s*="),
    "current_assign": re.compile(r"current_candle\s*="),
    "engine_df_assign": re.compile(r"\.df\s*="),
    "candle_engine_replace_history": re.compile(r"\bengine\.replace_history\s*\("),
}

ALLOWED_RELATIVE_FILES = {
    Path("data/candle_engine.py"),
}


def test_production_candle_engine_private_mutation_is_limited_to_candle_engine() -> (
    None
):
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(SRC_ROOT)
        if relative in ALLOWED_RELATIVE_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in MUTATION_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{relative}:{line}:{name}")
    assert offenders == []


def test_market_data_manager_ohlc_projection_is_refreshed_not_appended() -> None:
    path = SRC_ROOT / "data" / "market_data_manager.py"
    text = path.read_text(encoding="utf-8")
    assert "def _refresh_candle_projection" in text
    assert "_ohlc[self._bar_symbol_key(symbol)].append" not in text
    assert "ingest_historical_bar(row)" not in text
