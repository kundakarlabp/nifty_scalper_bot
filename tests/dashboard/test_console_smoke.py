from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]


def test_streamlit_console_renders_without_script_exception(monkeypatch) -> None:
    monkeypatch.setenv("BOT_ADMIN_API_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BOT_SERVICE_NAME", "niftybot-test")
    app = AppTest.from_file(str(ROOT / "dashboard" / "superlite_console.py"))
    app.run(timeout=15)
    assert not app.exception
    rendered = "\n".join(str(markdown.value) for markdown in app.markdown)
    assert "ADMIN API UNREACHABLE" in rendered
    assert "UNKNOWN" in rendered
