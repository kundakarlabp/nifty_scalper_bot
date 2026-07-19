from __future__ import annotations

import re
from pathlib import Path

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


def test_production_candle_engine_instantiation_is_limited_to_mdm_owner() -> None:
    offenders: list[str] = []
    allowed = {Path("data/market_data_manager.py")}
    pattern = re.compile(r"\bCandleEngine\s*\(")
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(SRC_ROOT)
        if relative in ALLOWED_RELATIVE_FILES or relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line}:candle_engine_instantiation")
    assert offenders == []


def test_rest_poll_path_does_not_construct_completed_ohlc_candle() -> None:
    text = (SRC_ROOT / "data" / "market_data_manager.py").read_text(encoding="utf-8")
    ingest_body = re.search(
        r"def ingest_rest_quote\(.*?\n    def ingest_historical_bar", text, re.S
    )
    assert ingest_body is not None
    assert "_process_poll_quote" not in text
    assert "_emit_poll_candle" not in text
    assert "ingest_historical_ohlc" not in ingest_body.group(0)
    assert "POLL CANDLE EMITTED" not in text


def test_strategy_runner_does_not_call_candle_engine_on_tick() -> None:
    text = (SRC_ROOT / "strategies" / "runner.py").read_text(encoding="utf-8")
    assert "from nifty_scalper_bot.data.candle_engine import CandleEngine" not in text
    assert "CandleEngine(" not in text


def test_live_production_callers_do_not_use_replace_history() -> None:
    offenders: list[str] = []
    pattern = re.compile(r"\.replace_history\s*\(")
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(SRC_ROOT)
        if relative == Path("data/candle_engine.py") or relative == Path(
            "strategies/indicators.py"
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 40) : match.start()]
            if "_indicator_engine" in prefix:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{relative}:{line}:replace_history")
    assert offenders == []


def test_completed_historical_writes_use_candle_engine_import_history() -> None:
    text = (SRC_ROOT / "data" / "market_data_manager.py").read_text(encoding="utf-8")
    assert "engine.import_history(frame" in text
    assert "_emit_poll_candle(candle)" not in text
