from pathlib import Path
import ast

path = Path("src/nifty_scalper_bot/execution/position_manager.py")
text = path.read_text(encoding="utf-8")
head = "\n".join(text.splitlines()[:40])
if "import math" not in head:
    anchor = "import json\n"
    if anchor not in text:
        raise RuntimeError("position manager import anchor missing")
    text = text.replace(anchor, anchor + "import math\n", 1)
    path.write_text(text, encoding="utf-8")
ast.parse(text, filename=str(path))
print("patched execution integration follow-up")
