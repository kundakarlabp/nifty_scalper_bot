"""Read-only, bounded and low-overhead Streamlit review console."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, time
from zoneinfo import ZoneInfo

import streamlit as st

from dashboard.superlite_events import csv_bytes, deduplicate_events, filter_events, parse_event

IST = ZoneInfo("Asia/Kolkata")
ADMIN_API = os.getenv("BOT_ADMIN_API_URL", "http://127.0.0.1:8081").rstrip("/")
ADMIN_PUBLIC = os.getenv("BOT_ADMIN_PUBLIC_URL", "http://15.206.3.6:8081/admin")
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
EVENT_TYPES = ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"]
PNL = re.compile(r"\bpnl=(-?\d+(?:\.\d+)?)", re.IGNORECASE)

st.set_page_config(page_title="Nifty Scalper Review", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--bg:#070b11;--panel:#0d151f;--line:#223249;--text:#e7edf5;--muted:#8292a7;--green:#39d98a;--amber:#f4c45d;--red:#ff6475;--blue:#67a9ff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)}
header,[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}.block-container{max-width:none!important;padding:.55rem .85rem 1rem!important}
.hero{background:#0d1d2c;border:1px solid #29445e;border-radius:12px;padding:11px 14px;margin-bottom:8px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.hero b{font-size:1.15rem}.muted{color:var(--muted);font-size:.72rem}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin:7px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px}.label{font-size:.61rem;color:var(--muted);text-transform:uppercase}.value{font-size:.95rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.feed{height:510px;overflow:auto;background:#05090e;border:1px solid var(--line);border-radius:10px}.row{display:grid;grid-template-columns:140px 65px minmax(0,1fr);gap:8px;padding:5px 8px;border-bottom:1px solid #152233;font:11.5px/1.4 ui-monospace,Consolas,monospace}.ts{color:#7f91a6}.msg{word-break:break-word}.ERROR{color:var(--red)}.WARNING{color:var(--amber)}.TRADE{color:#55dfd4}.SIGNAL{color:var(--blue)}.RISK{color:#c39aff}.SYSTEM{color:#aebdcc}@media(max-width:750px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.row{grid-template-columns:100px 55px minmax(0,1fr);font-size:9.5px}}
</style>
""", unsafe_allow_html=True)


def _get_json(path: str) -> dict:
    try:
        with urllib.request.urlopen(ADMIN_API + path, timeout=1.4) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else {}
    except (OSError, ValueError, urllib.error.URLError):
        return {}


@st.cache_data(ttl=6, show_spinner=False)
def status_snapshot() -> dict:
    return _get_json("/admin/api/status")


@st.cache_data(ttl=8, show_spinner=False)
def recent_events(lines: int = 650) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["journalctl", "-u", SERVICE, "-n", str(max(100, min(lines, 1000))), "--no-pager", "-o", "cat"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        raw = result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        raw = ""
    return deduplicate_events(event for line in raw.splitlines() if (event := parse_event(line)))


def card(label: str, value: object, css: str = "") -> str:
    return f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value {css}">{html.escape(str(value))}</div></div>'


def feed_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return '<div class="feed"><div class="row"><span></span><span></span><span class="muted">No matching actionable events.</span></div></div>'
    body = "".join(
        f'<div class="row"><span class="ts">{html.escape(row.get("timestamp_ist", ""))}</span>'
        f'<b class="{html.escape(row.get("type", "SYSTEM"))}">{html.escape(row.get("type", "SYSTEM"))}</b>'
        f'<span class="msg">{html.escape(row.get("message", ""))}</span></div>'
        for row in reversed(rows)
    )
    return f'<div class="feed">{body}</div>'


def pnl_summary(rows: list[dict[str, str]]) -> tuple[float, int]:
    values: list[float] = []
    for row in rows:
        if "BRACKET_CLOSED" not in row.get("message", ""):
            continue
        match = PNL.search(row["message"])
        if match:
            try: values.append(float(match.group(1)))
            except ValueError: pass
    return round(sum(values), 2), len(values)


now = datetime.now(IST)
within_session = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
st.markdown(
    f'<div class="hero"><div><b>⚡ Nifty Scalper Review</b><div class="muted">Read only · bounded journal reads · no background follower</div></div><div><b>{"SESSION OPEN" if within_session else "OUTSIDE SESSION"}</b><div class="muted">{now:%d %b %Y · %I:%M %p IST}</div></div></div>',
    unsafe_allow_html=True,
)
controls = st.columns([1, 1, 1.1, 2.2, .9, .9, 1.1], gap="small")
auto_refresh = controls[0].toggle("Auto-refresh", value=within_session)
refresh_seconds = controls[1].selectbox("Every", [8, 10, 15, 30], index=1, format_func=lambda value: f"{value}s")
event_type = controls[2].selectbox("Event", EVENT_TYPES)
query = controls[3].text_input("Contains", placeholder="symbol, blocker, order…")
row_limit = controls[4].selectbox("Rows", [50, 100, 200, 300], index=1)
trade_only = controls[5].toggle("Trade events", value=False)
controls[6].link_button("Open admin", ADMIN_PUBLIC, width="stretch")


def render() -> None:
    status = status_snapshot()
    rows_all = recent_events()
    rows = filter_events(rows_all, event_type, query)
    if trade_only:
        rows = [row for row in rows if row.get("type") == "TRADE"]
    rows = rows[-row_limit:]
    broker = status.get("broker") or {}
    recon = status.get("reconciliation") or {}
    selected = status.get("selected") or {}
    pnl_total, closed = pnl_summary(rows_all)
    st.markdown(
        '<div class="cards">'
        + card("Engine", "UP" if status.get("engine_loaded") else "DOWN", "ok" if status.get("engine_loaded") else "bad")
        + card("Operational", "READY" if status.get("operational_ready") else "BLOCKED", "ok" if status.get("operational_ready") else "warn")
        + card("Mode", status.get("mode") or "UNKNOWN", "ok" if status.get("live_orders_armed") else "warn")
        + card("Broker", "READY" if broker.get("ready") else "UNKNOWN", "ok" if broker.get("ready") else "warn")
        + card("Reconciled", "YES" if recon.get("completed") else "NO", "ok" if recon.get("completed") else "warn")
        + '</div><div class="cards">'
        + card("Visible events", len(rows))
        + card("Trade events", sum(row.get("type") == "TRADE" for row in rows))
        + card("Signal events", sum(row.get("type") == "SIGNAL" for row in rows))
        + card("Error events", sum(row.get("type") == "ERROR" for row in rows), "bad" if any(row.get("type") == "ERROR" for row in rows) else "")
        + card("Log realised P&L", f"₹{pnl_total:,.2f} · {closed} closed", "ok" if pnl_total >= 0 else "bad")
        + '</div><div class="cards">'
        + card("ATM", selected.get("atm") or "—")
        + card("Selected CE", selected.get("ce") or "—")
        + card("Selected PE", selected.get("pe") or "—")
        + card("Running", status.get("running") or "—")
        + card("Remote", status.get("remote") or "—", "warn" if status.get("stale") else "")
        + "</div>" + feed_html(rows),
        unsafe_allow_html=True,
    )
    st.download_button("Download current filtered events", csv_bytes(rows), file_name=f"niftybot-events-{datetime.now(IST):%Y%m%d-%H%M}.csv", mime="text/csv", width="stretch")
    with st.expander("Technical status", expanded=False):
        st.json(status)


if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_seconds}s")
    def live_fragment() -> None:
        render()
    live_fragment()
else:
    render()
