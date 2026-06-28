from __future__ import annotations

from scripts._execution_patch_utils import assert_parses, replace_once

POSITION = "src/nifty_scalper_bot/execution/position_manager.py"
WORKFLOW = ".github/workflows/live-exit-safety-ci.yml"

replace_once(POSITION, "import json\n", "import json\nimport math\n")

replace_once(
    WORKFLOW,
    "  test:\n    runs-on: ubuntu-latest\n",
    "  test:\n    runs-on: ubuntu-latest\n"
    "    env:\n"
    "      BRACKET_AUTO_RESTORE: \"false\"\n",
)
replace_once(
    WORKFLOW,
    "      - name: Entry recovery and fill accounting\n",
    "      - name: Canonical execution restart and exposure recovery\n"
    "        env:\n"
    "          BRACKET_AUTO_RESTORE: \"true\"\n"
    "        run: |\n"
    "          python -m pytest -q \\\n"
    "            tests/execution/test_canonical_execution_recovery.py \\\n"
    "            tests/execution/test_execution_safety_audit_fixes.py\n"
    "      - name: Entry recovery and fill accounting\n",
)

assert_parses(POSITION)
print("patched execution integration follow-ups")
