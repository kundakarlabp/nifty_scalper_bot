from pathlib import Path
import re

path = Path("src/nifty_scalper_bot/data/market_data_hardening.py")
text = path.read_text(encoding="utf-8")

loop_pattern = re.compile(
    r"(?P<prefix>\s+raw = q\.get\(timeout=0\.25\)\n"
    r"\s+except queue\.Empty:\n"
    r"\s+continue\n"
    r"\s+try:\n)"
    r"(?P<process>\s+self\._process_queued_tick\(raw\))"
)
text, loop_count = loop_pattern.subn(
    lambda match: (
        match.group("prefix")
        + '            if raw.get("_mdm_worker_stop_sentinel"):\n'
        + "                break\n"
        + match.group("process")
    ),
    text,
    count=1,
)
if loop_count != 1:
    raise SystemExit(f"worker loop anchor count={loop_count}")

stop_pattern = re.compile(
    r'(?P<prefix>\s+self\._tick_worker_state = "STOPPING"\n'
    r"\s+self\._tick_worker_stop\.set\(\)\n)"
    r"(?P<next>\s+if thread is threading\.current_thread\(\):)"
)
text, stop_count = stop_pattern.subn(
    lambda match: (
        match.group("prefix")
        + "        try:\n"
        + "            self._fallback_tick_queue.put_nowait(\n"
        + '                {"_mdm_worker_stop_sentinel": True}\n'
        + "            )\n"
        + "        except queue.Full:\n"
        + "            pass\n"
        + match.group("next")
    ),
    text,
    count=1,
)
if stop_count != 1:
    raise SystemExit(f"worker stop anchor count={stop_count}")

path.write_text(text, encoding="utf-8")
