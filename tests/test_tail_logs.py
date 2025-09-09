import os
import tempfile

from src.server.logging_utils import _tail_logs


def test_tail_logs_returns_last_n_lines():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        path = f.name
        for i in range(1000):
            f.write(f"line-{i}\n")
    try:
        out = _tail_logs(path, n=3)
        assert out == ["line-997", "line-998", "line-999"]
    finally:
        os.remove(path)


def test_tail_logs_handles_missing_file(caplog):
    with caplog.at_level("WARNING", logger="main"):
        result = _tail_logs("/non/existent/path.log", n=3)
    assert result == []
    assert "tail_logs failed for" in caplog.text
