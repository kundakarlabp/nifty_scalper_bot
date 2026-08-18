from __future__ import annotations

from pathlib import Path

APP = Path("src/nifty_scalper_bot/core/app.py")
text = APP.read_text(encoding="utf-8")

marker = '''def _as_bool(value: object, default: bool = False) -> bool:\n'''
helper = '''_READINESS_MISSING_DIAGNOSTIC_LABELS = {\n    "options_ce": "ce_live_tick_readiness_insufficient",\n    "options_pe": "pe_live_tick_readiness_insufficient",\n}\n\n\ndef _readiness_missing_diagnostics(missing: Iterable[object]) -> list[str]:\n    """Describe legacy MDM readiness keys without implying contracts are absent."""\n    return [\n        _READINESS_MISSING_DIAGNOSTIC_LABELS.get(str(item), str(item))\n        for item in missing\n    ]\n\n\n'''
if helper not in text:
    assert text.count(marker) == 1, "unexpected _as_bool marker count"
    text = text.replace(marker, helper + marker, 1)

old = '''                                LOGGER.info(\n                                    "DATA_PIPELINE_NOT_READY hard_ready=%s spot_ready=%s missing=%s",\n                                    hard_ready,\n                                    spot_ready,\n                                    missing_hard,\n'''
new = '''                                LOGGER.info(\n                                    "DATA_PIPELINE_NOT_READY hard_ready=%s spot_ready=%s missing_live_tick=%s",\n                                    hard_ready,\n                                    spot_ready,\n                                    _readiness_missing_diagnostics(missing_hard),\n'''
assert text.count(old) == 1, "unexpected DATA_PIPELINE_NOT_READY block count"
text = text.replace(old, new, 1)

old = '''                                    LOGGER.error(\n                                        "startup_pipeline_incomplete missing=%s",\n                                        (\n                                            ",".join(missing_hard)\n                                            if missing_hard\n                                            else "unknown"\n                                        ),\n'''
new = '''                                    LOGGER.error(\n                                        "startup_pipeline_incomplete missing_live_tick=%s",\n                                        (\n                                            ",".join(\n                                                _readiness_missing_diagnostics(missing_hard)\n                                            )\n                                            if missing_hard\n                                            else "unknown"\n                                        ),\n'''
assert text.count(old) == 1, "unexpected startup_pipeline_incomplete block count"
text = text.replace(old, new, 1)

old = '''                                    LOGGER.error(\n                                        "LIVE_TRADING_BLOCKED reason=startup_pipeline_incomplete missing=%s",\n                                        missing_hard,\n'''
new = '''                                    LOGGER.error(\n                                        "LIVE_TRADING_BLOCKED reason=startup_pipeline_incomplete missing_live_tick=%s",\n                                        _readiness_missing_diagnostics(missing_hard),\n'''
assert text.count(old) == 1, "unexpected LIVE_TRADING_BLOCKED block count"
text = text.replace(old, new, 1)

APP.write_text(text, encoding="utf-8")
