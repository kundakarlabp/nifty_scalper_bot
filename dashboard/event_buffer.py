"""Bounded, event-only systemd journal tail for the Streamlit console."""
from __future__ import annotations
import re
import subprocess
import threading
import time
from collections import deque

STAMP = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST)\]")
EVENTS = ("SIGNAL","ORDER","FILLED","ENTRY","EXIT","TARGET","STOP","PNL","POSITION",
          "BROKER","AUTH","READY","BLOCKER","RISK","COOLDOWN","ERROR","FAIL","WARN",
          "DEGRADED","OPERATIONAL","STARTUP","SHUTDOWN")
IGNORE = ("HEARTBEAT","POLLING","INDICATOR","NO SIGNAL","MARKET CLOSED","OUTSIDE SESSION")

def parse_event(line: str):
    match = STAMP.search(line)
    if not match:
        return None
    message = line[match.end():].strip()
    upper = message.upper()
    if any(x in upper for x in IGNORE) or not any(x in upper for x in EVENTS):
        return None
    if any(x in upper for x in ("ERROR","FAIL","TRACEBACK","REJECTED")):
        kind = "ERROR"
    elif any(x in upper for x in ("WARN","DEGRADED")):
        kind = "WARNING"
    elif any(x in upper for x in ("ORDER","FILLED","ENTRY","EXIT","TARGET","STOP","PNL","POSITION")):
        kind = "TRADE"
    elif "SIGNAL" in upper:
        kind = "SIGNAL"
    elif any(x in upper for x in ("RISK","COOLDOWN")):
        kind = "RISK"
    else:
        kind = "SYSTEM"
    return {"timestamp_ist": match.group(1), "type": kind, "message": message}

class EventRing:
    def __init__(self, service: str, max_events: int = 3000):
        self.service = service
        self.rows = deque(maxlen=max_events)
        self.lock = threading.RLock()
        self.connected = False
        self.last_event = 0.0
        self.restarts = 0
        threading.Thread(target=self._run, daemon=True, name="journal-event-tail").start()

    def _run(self):
        while True:
            proc = None
            try:
                proc = subprocess.Popen(
                    ["journalctl","-u",self.service,"-n","1200","-f","--no-pager","-o","cat"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
                self.connected = True
                for line in proc.stdout or ():
                    event = parse_event(line)
                    if event:
                        with self.lock:
                            if not self.rows or self.rows[-1] != event:
                                self.rows.append(event)
                                self.last_event = time.time()
            except Exception:
                pass
            finally:
                self.connected = False
                self.restarts += 1
                if proc and proc.poll() is None:
                    proc.terminate()
                time.sleep(1.5)

    def snapshot(self):
        with self.lock:
            return list(self.rows)

    def stats(self):
        with self.lock:
            return {"connected": self.connected, "last_event": self.last_event,
                    "restarts": self.restarts, "size": len(self.rows)}
