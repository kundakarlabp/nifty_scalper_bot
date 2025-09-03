from pathlib import Path
from src.backtesting.data_source import load_and_prepare_data
from src.config import settings

def test_load_and_prepare_synth_when_rows_insufficient(tmp_path: Path) -> None:
    csv = tmp_path / "tiny.csv"
    csv.write_text("open,high,low,close,volume\n1,2,0,1,100\n")
    df = load_and_prepare_data(csv)
    assert len(df) >= settings.strategy.min_bars_for_signal
