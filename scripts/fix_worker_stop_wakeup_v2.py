from pathlib import Path

path = Path("src/nifty_scalper_bot/data/market_data_hardening.py")
text = path.read_text(encoding="utf-8")

old_loop = '''        try:
            raw = q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            self._process_queued_tick(raw)
'''
new_loop = '''        try:
            raw = q.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            if raw.get("_mdm_worker_stop_sentinel"):
                break
            self._process_queued_tick(raw)
'''
if text.count(old_loop) != 1:
    raise SystemExit(f"worker loop anchor count={text.count(old_loop)}")
text = text.replace(old_loop, new_loop, 1)

old_stop = '''            self._tick_worker_state = "STOPPING"
            self._tick_worker_stop.set()
        if thread is threading.current_thread():
'''
new_stop = '''            self._tick_worker_state = "STOPPING"
            self._tick_worker_stop.set()
            try:
                self._fallback_tick_queue.put_nowait(
                    {"_mdm_worker_stop_sentinel": True}
                )
            except queue.Full:
                pass
        if thread is threading.current_thread():
'''
if text.count(old_stop) != 1:
    raise SystemExit(f"worker stop anchor count={text.count(old_stop)}")
text = text.replace(old_stop, new_stop, 1)

path.write_text(text, encoding="utf-8")
