"""Fast event-only Streamlit console for the Nifty scalper."""
from __future__ import annotations

import csv
import html
import io
import json
import os
import subprocess
import time as clock
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from event_buffer import EventRing, parse_event

st.set_page_config(page_title="Nifty Scalper Live", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container{max-width:1660px;padding-top:.5rem}.stMetric{border:1px solid #26384d;border-radius:13px;padding:.5rem .7rem;background:#0d1722}
.hero{display:flex;justify-content:space-between;align-items:center;gap:10px;background:linear-gradient(135deg,#0b1b2b,#10283b);border:1px solid #27415a;border-radius:15px;padding:13px 15px;margin-bottom:8px}
.title{font-size:1.3rem;font-weight:750}.sub,.muted{color:#8ea0b4;font-size:.78rem}.pill{border:1px solid #29405a;border-radius:999px;padding:5px 10px;font-size:.75rem;font-weight:700}
.feed{height:520px;overflow:auto;background:#050a10;border:1px solid #213247;border-radius:13px;padding:8px 10px;font:11.5px/1.45 monospace}
.ev{display:grid;grid-template-columns:145px 72px 1fr;gap:7px;padding:5px 3px;border-bottom:1px solid #1b2938}.ts{color:#718399}.msg{color:#cbd7e3;word-break:break-word}
.ERROR{color:#ff5d6c}.WARNING{color:#ffc857}.TRADE{color:#58e6d9}.SIGNAL{color:#5da9ff}.RISK{color:#b28cff}.SYSTEM{color:#aebdcc}
.ok,.warn,.bad{padding:8px 10px;border-radius:8px;margin:6px 0 10px}.ok{border-left:4px solid #27d980;background:#122a22}.warn{border-left:4px solid #ffc857;background:#292415}.bad{border-left:4px solid #ff5d6c;background:#2b151a}
@media(max-width:760px){.hero{align-items:flex-start;flex-direction:column}.feed{height:440px;font-size:10px}.ev{grid-template-columns:116px 60px 1fr}}
</style>""", unsafe_allow_html=True)

IST = ZoneInfo("Asia/Kolkata")
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
API = os.getenv("BOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
APP = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
UPDATE_FILE = APP / "data" / "auto_update_status.json"


@st.cache_resource
def session() -> requests.Session:
    return requests.Session()


@st.cache_resource
def ring() -> EventRing:
    return EventRing(SERVICE, max_events=3000)


def get_json(path: str) -> dict[str, Any] | None:
    try:
        response = session().get(API + path, timeout=2)
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def select(rows: list[dict[str, str]], kind: str, query: str) -> list[dict[str, str]]:
    if kind != "ALL":
        rows = [row for row in rows if row["type"] == kind]
    needle = query.strip().lower()
    return [row for row in rows if needle in row["message"].lower()] if needle else rows


def feed(rows: list[dict[str, str]]) -> str:
    if not rows:
        return '<div class="feed"><span class="muted">Waiting for actionable events…</span></div>'
    lines = []
    for row in reversed(rows):
        lines.append(
            '<div class="ev">'
            f'<span class="ts">{html.escape(row["timestamp_ist"])}</span>'
            f'<b class="{row["type"]}">{row["type"]}</b>'
            f'<span class="msg">{html.escape(row["message"])}</span></div>'
        )
    return '<div class="feed">' + "".join(lines) + "</div>"


def read_history(day: date, start_at: time, end_at: time) -> tuple[list[dict[str, str]], str | None]:
    command = ["journalctl", "-u", SERVICE, "--since", f"{day} {start_at:%H:%M:%S}",
               "--until", f"{day} {end_at:%H:%M:%S}", "--no-pager", "-o", "cat"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except Exception as exc:
        return [], str(exc)
    if result.returncode:
        return [], result.stderr.strip() or "journal query failed"
    return [event for line in result.stdout.splitlines() if (event := parse_event(line))], None


def csv_data(rows: list[dict[str, str]]) -> bytes:
    target = io.StringIO()
    writer = csv.DictWriter(target, fieldnames=["timestamp_ist", "type", "message"])
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")


def updater() -> dict[str, Any]:
    try:
        value = json.loads(UPDATE_FILE.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(APP), "rev-parse", "--short", ref],
                                       text=True, timeout=2).strip()
    except Exception:
        return "—"


def status_panel() -> None:
    livez, readyz, trading = get_json("/livez"), get_json("/readyz"), get_json("/health/trading")
    broker = (trading or {}).get("broker") or {}
    recon = (trading or {}).get("reconciliation") or {}
    blockers = (trading or {}).get("blockers") or (readyz or {}).get("blockers") or []
    cards = st.columns(8)
    values = [
        ("Process", "UP" if livez else "DOWN"),
        ("Engine", "LOADED" if (livez or {}).get("bot_loaded") else "STARTING"),
        ("Execution", "READY" if (trading or {}).get("ready") else "BLOCKED"),
        ("Orders", "ARMED" if (trading or {}).get("live_orders_armed") else "OFF"),
        ("Broker", "READY" if broker.get("ready") else "NOT READY"),
        ("Balance", str(broker.get("balance") or "—")),
        ("Reconciled", "YES" if recon.get("completed") else "NO"),
        ("Auth", "INVALID" if broker.get("auth_invalid") else "OK"),
    ]
    for card, value in zip(cards, values):
        card.metric(*value)
    if blockers:
        st.markdown('<div class="warn"><b>Blockers:</b> ' + " • ".join(map(html.escape, map(str, blockers))) + "</div>", unsafe_allow_html=True)
    elif livez:
        st.markdown('<div class="ok"><b>Operational checks passed.</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bad"><b>Bot API unreachable.</b></div>', unsafe_allow_html=True)
    st.session_state["diag"] = {"livez": livez, "readyz": readyz, "trading": trading}


now = datetime.now(IST)
is_open = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
market = "OPEN" if is_open else "CLOSED"
st.markdown(f'<div class="hero"><div><div class="title">⚡ Nifty Scalper Live Console</div>'
            f'<div class="sub">Event-only • bounded memory • read-only • optimized for market hours</div></div>'
            f'<div><span class="pill">{market}</span> <span class="pill">{now:%d %b · %I:%M:%S %p IST}</span></div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Live feed")
    live = st.toggle("Live refresh", True)
    interval = st.select_slider("Refresh seconds", [1, 2, 3, 5, 10], value=2)
    kind = st.selectbox("Event type", ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"])
    query = st.text_input("Search", placeholder="NIFTY, order, blocker…")
    limit = st.select_slider("Events shown", [50, 100, 200, 300, 500], value=200)
    if st.button("Refresh now", use_container_width=True):
        st.rerun()
    st.divider()
    state = updater()
    st.caption(f"Running: `{git_commit('HEAD')}`")
    st.caption(f"Fetched main: `{git_commit('origin/main')}`")
    st.caption(state.get("message", "Auto-updater not yet reporting"))

st.subheader("Trading status")
if hasattr(st, "fragment"):
    @st.fragment(run_every="3s")
    def status_fragment() -> None:
        status_panel()
    status_fragment()
else:
    status_panel()

events = ring()


def live_panel() -> None:
    rows = select(events.snapshot(), kind, query)[-limit:]
    stats = events.stats()
    cards = st.columns(6)
    totals = [("Visible", len(rows)), ("Trades", sum(r["type"] == "TRADE" for r in rows)),
              ("Signals", sum(r["type"] == "SIGNAL" for r in rows)),
              ("Risk", sum(r["type"] == "RISK" for r in rows)),
              ("Errors", sum(r["type"] == "ERROR" for r in rows)),
              ("Last event", f"{int(clock.time()-stats['last_event'])}s ago" if stats['last_event'] else "waiting")]
    for card, value in zip(cards, totals):
        card.metric(*value)
    st.markdown(feed(rows), unsafe_allow_html=True)
    st.caption(f"Follower: {'connected' if stats['connected'] else 'reconnecting'} • buffer {stats['size']:,} • restarts {stats['restarts']}")


st.subheader("Actionable event feed")
if live and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{interval}s")
    def event_fragment() -> None:
        live_panel()
    event_fragment()
else:
    live_panel()

st.divider()
st.subheader("Event history and export")
cols = st.columns([1.2, 1, 1, 1, 1.4])
day = cols[0].date_input("Date", date.today(), max_value=date.today())
start_at = cols[1].time_input("From", time(9, 15), step=60)
end_at = cols[2].time_input("To", time(15, 30), step=60)
hkind = cols[3].selectbox("Type", ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"], key="hkind")
hquery = cols[4].text_input("Contains", key="hquery")
load_col, clear_col = st.columns(2)
load = load_col.button("Load selected history", type="primary", use_container_width=True)
if clear_col.button("Clear history", use_container_width=True):
    st.session_state.pop("history", None)
    st.session_state.pop("history_error", None)

if start_at >= end_at:
    st.error("From must be earlier than To.")
elif load:
    with st.spinner("Reading selected journal window…"):
        st.session_state["history"], st.session_state["history_error"] = read_history(day, start_at, end_at)

if st.session_state.get("history_error"):
    st.error(st.session_state["history_error"])
elif st.session_state.get("history"):
    rows = select(st.session_state["history"], hkind, hquery)
    table, preview = st.tabs(["Trading table", "Preview"])
    with table:
        st.dataframe(pd.DataFrame(rows, columns=["timestamp_ist", "type", "message"]),
                     use_container_width=True, hide_index=True, height=340)
    with preview:
        st.markdown(feed(rows[-500:]), unsafe_allow_html=True)
    filename = f"niftybot-events-{day}-{start_at:%H%M}-{end_at:%H%M}.csv"
    st.download_button(f"⬇️ Download {len(rows):,} events", csv_data(rows), filename,
                       "text/csv", type="primary", use_container_width=True)
else:
    st.info("Select a time window and press **Load selected history**.")

with st.expander("Diagnostics"):
    st.json({**st.session_state.get("diag", {}), "updater": updater()})
