from __future__ import annotations

from scripts._execution_patch_utils import assert_parses, replace_once

POSITION = "src/nifty_scalper_bot/execution/position_manager.py"

replace_once(POSITION, "import json\n", "import json\nimport math\n")

assert_parses(POSITION)
print("patched execution integration follow-up")
