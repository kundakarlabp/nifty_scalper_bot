from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


# 1) Signal owns deterministic identity natively.
path = "src/nifty_scalper_bot/strategies/signal_generator.py"
text = read(path)
pattern = re.compile(
    r"    @property\n    def deterministic_id\(self\) -> str:\n"
    r"        \"\"\"\n"
    r"        Generate a stable, restart-safe ID for this signal\.\n"
    r"        Logic: HASH\(Strategy \+ Symbol \+ Action \+ MinuteTimestamp\)\n"
    r"        \"\"\"\n"
    r".*?"
    r"        return hashlib\.md5\(raw_sig\.encode\(\)\)\.hexdigest\(\)\[:16\]\n",
    re.DOTALL,
)
replacement = '''    @property
    def deterministic_id(self) -> str:
        """Return the native stable setup identity for this signal."""
        from nifty_scalper_bot.strategies.signal_identity import deterministic_signal_id

        return deterministic_signal_id(self)
'''
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise RuntimeError(f"signal_generator deterministic_id: expected one replacement, got {count}")
write(path, text)

# 2) EliteStrategy owns final signal observability in its canonical generate path.
path = "src/nifty_scalper_bot/strategies/elite_strategies/base_elite.py"
text = read(path)
text = replace_once(
    text,
    "from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy\n",
    "from nifty_scalper_bot.strategies.signal_generator import Signal, Strategy\n"
    "from nifty_scalper_bot.strategies.signal_identity import finalize_signal_observability\n",
    label="base_elite import",
)
text = replace_once(
    text,
    "                return self._process_signal(elite_signal)\n\n            LOGGER.debug(\n",
    "                processed_signal = self._process_signal(elite_signal)\n"
    "                return finalize_signal_observability(\n"
    "                    processed_signal,\n"
    "                    indicators_payload,\n"
    "                    strategy_name=self.name,\n"
    "                    symbol=symbol,\n"
    "                )\n\n"
    "            LOGGER.debug(\n",
    label="base_elite native finalization",
)
write(path, text)

# 3) IndicatorEngine owns runtime-context normalization natively.
path = "src/nifty_scalper_bot/strategies/indicators.py"
text = read(path)
text = replace_once(
    text,
    "from nifty_scalper_bot.utils.logging import get_logger, log_throttled\n",
    "from nifty_scalper_bot.strategies.runtime_context_contract import (\n"
    "    normalise_live_direction_context,\n"
    ")\n"
    "from nifty_scalper_bot.utils.logging import get_logger, log_throttled\n",
    label="indicators runtime context import",
)
text = replace_once(
    text,
    "                for key, value in dict(context).items():\n"
    "                    if key in allowed_context_keys:\n"
    "                        symbol_context[key] = value\n"
    "                self._runtime_context[symbol] = symbol_context\n",
    "                for key, value in dict(context).items():\n"
    "                    if key in allowed_context_keys:\n"
    "                        symbol_context[key] = value\n"
    "                symbol_context.update(normalise_live_direction_context(context))\n"
    "                self._runtime_context[symbol] = symbol_context\n",
    label="indicators native runtime normalization",
)
write(path, text)

# 4) runtime_context_contract becomes pure helpers; remove installer mutation.
path = "src/nifty_scalper_bot/strategies/runtime_context_contract.py"
text = read(path)
marker = "\ndef install_indicator_runtime_context_contract() -> bool:\n"
if marker not in text:
    raise RuntimeError("runtime_context_contract installer marker missing")
text = text.split(marker, 1)[0].rstrip() + "\n"
write(path, text)

# 5) Move all consumers from the patch module to the neutral helper module and
# use public helper names. This includes execution callers and tests so the
# strategy package no longer needs a compatibility import hook.
for root in (ROOT / "src", ROOT / "tests"):
    for file in root.rglob("*.py"):
        if file.name == "signal_identity_patch.py":
            continue
        body = file.read_text(encoding="utf-8")
        updated = body.replace(
            "nifty_scalper_bot.strategies.signal_identity_patch",
            "nifty_scalper_bot.strategies.signal_identity",
        )
        updated = re.sub(r"\b_deterministic_id\b", "deterministic_signal_id", updated)
        updated = re.sub(r"\b_stamp_evaluation_identity\b", "stamp_evaluation_identity", updated)
        updated = re.sub(r"\b_option_thesis\b", "option_thesis", updated)
        if updated != body:
            file.write_text(updated, encoding="utf-8")

old_patch = ROOT / "src/nifty_scalper_bot/strategies/signal_identity_patch.py"
if not old_patch.exists():
    raise RuntimeError("signal_identity_patch.py missing before migration")
old_patch.unlink()

# 6) Rewrite tests that explicitly asserted import-time patch installation.
write(
    "tests/strategies/test_identity_contract_installed.py",
    '''"""Identity/runtime contracts are native owners, not import-time patches."""

from __future__ import annotations

import inspect

import nifty_scalper_bot.strategies as strategies
from nifty_scalper_bot.strategies.elite_strategies.base_elite import EliteStrategy
from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.strategies.signal_generator import Signal


def test_signal_identity_is_native_property() -> None:
    source = inspect.getsource(Signal.deterministic_id.fget)
    assert "deterministic_signal_id" in source
    assert isinstance(Signal.deterministic_id, property)
    assert not hasattr(Signal, "_stable_setup_identity_patch")


def test_elite_signal_finalization_is_native() -> None:
    source = inspect.getsource(EliteStrategy.generate_signal)
    assert "finalize_signal_observability" in source
    assert "_elite_signal_observability_patch" not in source


def test_runtime_context_normalization_is_native() -> None:
    source = inspect.getsource(IndicatorEngine.set_runtime_context)
    assert "normalise_live_direction_context" in source
    assert "_live_direction_contract_installed" not in source


def test_strategy_package_has_no_runtime_patch_installer() -> None:
    source = inspect.getsource(strategies)
    assert "apply_patches" not in source
    assert "install_indicator_runtime_context_contract" not in source


def test_elite_exports_are_available() -> None:
    assert strategies.__all__
''',
)

path = "tests/strategies/test_runtime_context_contract_autoinstall.py"
if (ROOT / path).exists():
    body = read(path)
    body = body.replace(
        "def test_strategy_package_installs_quote_context_contract() -> None:",
        "def test_indicator_engine_natively_preserves_quote_context_contract() -> None:",
    )
    body = body.replace(
        '    importlib.import_module("nifty_scalper_bot.strategies")\n',
        '    importlib.import_module("nifty_scalper_bot.strategies.indicators")\n',
    )
    write(path, body)

# Hard assertions: no strategy package mutation installer remains.
init_source = read("src/nifty_scalper_bot/strategies/__init__.py")
if "apply_patches" in init_source or "install_indicator_runtime_context_contract" in init_source:
    raise RuntimeError("strategy package still installs runtime patches")
for root in (ROOT / "src", ROOT / "tests"):
    for file in root.rglob("*.py"):
        if "signal_identity_patch" in file.read_text(encoding="utf-8"):
            raise RuntimeError(f"stale signal_identity_patch reference: {file}")
