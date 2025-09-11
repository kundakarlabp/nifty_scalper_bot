import csv
from pathlib import Path

from src.diagnostics.metrics import daily_summary, record_trade


def test_journal_row_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    record_trade(1.0, 2.0)
    record_trade(-0.5, 1.0)
    journal_dir = Path("data/journal")
    files = list(journal_dir.glob("*.csv"))
    assert files
    with files[0].open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["ts", "pnl_R", "slippage_bps"]
    assert len(rows) == 2
    summary = daily_summary()
    assert summary["trades"] == 2
    assert summary["R"] == 0.5
