from pathlib import Path

path = Path("src/nifty_scalper_bot/strategies/indicators.py")
text = path.read_text(encoding="utf-8")
block = '''\n\n# ── Explicit runtime-context contract integration (definition site) ─────────\n# Previously installed from strategies/__init__.py; centralized here so\n# IndicatorEngine always carries the contract regardless of import path.\nfrom nifty_scalper_bot.strategies.runtime_context_contract import (  # noqa: E402\n    install_indicator_runtime_context_contract as _install_ctx_contract,\n)\n\n_install_ctx_contract()\n'''
if text.count(block) != 1:
    raise RuntimeError("expected exactly one residual IndicatorEngine installer block")
path.write_text(text.replace(block, "\n", 1), encoding="utf-8")
