from pathlib import Path


def test_production_startup_imports_real_zerodha_not_legacy_dummy() -> None:
    source = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")
    assert (
        "from nifty_scalper_bot.data.rest.zerodha_client import ZerodhaKiteClient"
        in source
    )
    legacy_import = "from nifty_scalper_bot.bot.components import ZerodhaKiteClient"
    assert legacy_import not in source
