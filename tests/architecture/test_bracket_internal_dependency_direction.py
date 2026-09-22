from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXECUTION = ROOT / "src" / "nifty_scalper_bot" / "execution"
STAGED_BRACKET_MODULES = (
    "hardened_bracket_manager.py",
    "canonical_bracket_manager.py",
    "ledger_bracket_manager.py",
    "runtime_bracket_manager.py",
)


def test_staged_bracket_modules_depend_on_core_not_public_facade() -> None:
    public_import = (
        "from nifty_scalper_bot.execution import bracket_manager as _legacy"
    )
    core_import = "from . import bracket_core as _legacy"

    for filename in STAGED_BRACKET_MODULES:
        source = (EXECUTION / filename).read_text(encoding="utf-8")
        assert public_import not in source, filename
        assert core_import in source, filename
