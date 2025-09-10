from pathlib import Path

from src.backtesting.data_source import load_and_prepare_data
from src.config import settings


def test_adds_volume_column_when_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings.strategy, "min_bars_for_signal", 1, raising=False)
    csv = tmp_path / "no_volume.csv"
    rows = "\n".join(
        f"2020-01-01T00:{i:02d}:00,1,2,0,1" for i in range(50)
    )
    csv.write_text("datetime,open,high,low,close\n" + rows + "\n")
    df = load_and_prepare_data(csv)
    assert "volume" in df.columns
    assert (df["volume"] == 0).all()
