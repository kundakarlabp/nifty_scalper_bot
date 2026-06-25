"""Super-light read-only Streamlit operations console for AWS Lightsail."""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import time as clock
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from dashboard.event_buffer import EventRing
from dashboard.log_export import (
    csv_bytes,
    filter_events,
    read_actionable_events,
    read_raw_logs,
)

st.set_page_config(
    page_title="Nifty Scalper Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root{--bg:#070b11;--panel:#0d141e;--panel2:#101b28;--line:#213247;--text:#e8eef6;
--muted:#8797aa;--green:#3ddc97;--amber:#f5c451;--red:#ff6374;--blue:#65a9ff;--cyan:#55dfd4}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"],
[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="stAppViewBlockContainer"],.block-container{max-width:none!important;width:100%!important;
padding:.55rem .8rem 1.1rem!important}
.hero{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;
background:linear-gradient(135deg,#0b1a29,#10283a);border:1px solid #29445e;border-radius:14px;
padding:12px 15px;margin-bottom:9px;box-shadow:0 8px 26px rgba(0,0,0,.18)}
.brand{font-size:1.25rem;font-weight:800}.sub{font-size:.72rem;color:#91a3b7;margin-top:2px}
.pills{display:flex;gap:7px;flex-wrap:wrap}.pill{border:1px solid #35516d;border-radius:999px;
padding:4px 9px;font-size:.68rem;font-weight:800;background:#0b1723;color:#bed0e2}
.pill.good{color:#6ce9aa;border-color:#286c4a;background:#0e241b}.pill.warn{color:#f5d47f;border-color:#755f29;background:#2a2413}
.section{font-size:.9rem;font-weight:780;margin:3px 0 6px}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin-bottom:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:9px 10px;min-width:0}
.label{font-size:.62rem;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.value{font-size:1rem;font-weight:800;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.good-text{color:var(--green)!important}.warn-text{color:var(--amber)!important}.bad-text{color:var(--red)!important}.info-text{color:var(--blue)!important}
.alert{border-radius:9px;padding:8px 10px;font-size:.73rem;line-height:1.35;margin-bottom:8px}
.alert.ok{border-left:4px solid var(--green);background:#10271e;color:#c9f5dc}.alert.warn{border-left:4px solid var(--amber);background:#282313;color:#f5df9f}
.alert.bad{border-left:4px solid var(--red);background:#2b141a;color:#ffc3ca}
.feed-shell{background:#05090e;border:1px solid var(--line);border-radius:11px;overflow:hidden}
.feed-head,.ev{display:grid;grid-template-columns:142px 68px minmax(0,1fr);gap:8px}
.feed-head{padding:7px 9px;background:#0b141e;border-bottom:1px solid var(--line);font-size:.6rem;
text-transform:uppercase;letter-spacing:.07em;color:#8093a8}.feed{height:520px;overflow-y:auto;overscroll-behavior:contain}
.ev{padding:6px 9px;border-bottom:1px solid #142130;font:12px/1.42 ui-monospace,SFMono-Regular,Consolas,monospace}
.ev:hover{background:#0a121b}.ts{color:#8093a8}.msg{color:#d8e3ef;word-break:break-word}.badge{font-weight:850;font-size:10.5px}
.ERROR{color:var(--red)}.WARNING{color:var(--amber)}.TRADE{color:var(--cyan)}.SIGNAL{color:var(--blue)}.RISK{color:#c098ff}.SYSTEM{color:#aebdcc}
.empty{padding:22px;color:var(--muted);font:12px ui-monospace,monospace}.foot{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;
padding:6px 9px;border-top:1px solid var(--line);background:#09111a;color:#718399;font-size:.64rem}
.status-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.status-item{background:#09111a;border:1px solid #1d2b3a;border-radius:8px;padding:7px}
.status-key{font-size:.59rem;text-transform:uppercase;color:#74869a;letter-spacing:.06em}.status-val{font-size:.84rem;font-weight:800;margin-top:2px;overflow:hidden;text-overflow:ellipsis}
.deploy{display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #172536;padding:4px 0;font-size:.68rem}.deploy:last-child{border:0}
.deploy span:first-child{color:var(--muted)}.deploy span:last-child{font-family:ui-monospace,monospace;color:#c9d5e2;text-align:right}
[data-testid="stWidgetLabel"] p{font-size:.68rem!important;color:#8fa1b5!important}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stDateInput"] input{background:#09111a!important;border-color:#26384b!important}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:11px!important;background:var(--panel)}
.stButton>button,.stDownloadButton>button{border-radius:8px!important;font-weight:740!important}
@media(max-width:1000px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}.feed{height:470px}}
@media(max-width:700px){[data-testid="stAppViewBlockContainer"],.block-container{padding:.4rem!important}.cards{grid-template-columns:repeat(2,minmax(0,1fr))}
.feed-head,.ev{grid-template-columns:104px 53px minmax(0,1fr)}.ev{font-size:9.5px;padding:5px}.feed{height:430px}.brand{font-size:1.05rem}}
</style>
""",
    unsafe_allow_html=True,
)

IST = ZoneInfo("Asia/Kolkata")
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
API = os.getenv("BOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
APP = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
UPDATE_FILE = APP / "data" / "auto_update_status.json"
EVENT_TYPES = ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"]
TRADE_MARKERS = (
    "ORDER_SENT", "SENDING ORDER", "FILLED", "AVERAGE_PRICE", "EXIT", "TRADE_ATTEMPT",
    "ORDER_REJECTED", "BRACKET_EXIT", "BRACKET_CLOSED", "PNL", "SIGNAL_EXECUTION_RESULT",
)
PNL_RE = re.compile(r"\bpnl=(-?\d+(?:\.\d+)?)", re.IGNORECASE)


@st.cache_resource
def http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Connection": "keep-alive"})
    return session


@st.cache_resource
def event_ring() -> EventRing:
    return EventRing(SERVICE, max_events=2500)


@st.cache_data(ttl=2.5, show_spinner=False)
def get_json(path: str) -> dict[str, Any] | None:
    try:
        response = http_session().get(API + path, timeout=(0.35, 0.9))
        value = response.json()
        if not isinstance(value, dict):
            return None
        value = dict(value)
        value["_http_status"] = response.status_code
        return value
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(ttl=45, show_spinner=False)
def git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(APP), "rev-parse", "--short", ref],
            text=True,
            timeout=1.5,
        ).strip()
    except Exception:
        return "—"


def updater_state() -> dict[str, Any]:
    try:
        value = json.loads(UPDATE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def trades_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if any(marker in row.get("message", "").upper() for marker in TRADE_MARKERS)]


def realized_pnl(rows: list[dict[str, str]]) -> tuple[float, int, int]:
    values: list[float] = []
    for row in rows:
        message = row.get("message", "")
        if "BRACKET_CLOSED" not in message:
            continue
        match = PNL_RE.search(message)
        if match:
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return round(sum(values), 2), len(values), sum(value >= 0 for value in values)


def render_feed(rows: list[dict[str, str]]) -> str:
    if not rows:
        body = '<div class="empty">No matching actionable events.</div>'
    else:
        body = "".join(
            '<div class="ev">'
            f'<span class="ts">{html.escape(row.get("timestamp_ist", ""))}</span>'
            f'<span class="badge {html.escape(row.get("type", "SYSTEM"))}">{html.escape(row.get("type", "SYSTEM"))}</span>'
            f'<span class="msg">{html.escape(row.get("message", ""))}</span></div>'
            for row in reversed(rows)
        )
    return (
        '<div class="feed-shell"><div class="feed-head"><span>Time · IST</span>'
        f'<span>Event</span><span>Message</span></div><div class="feed">{body}</div></div>'
    )


def card(label: str, value: Any, css: str = "") -> str:
    return (
        '<div class="card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value {css}">{html.escape(str(value))}</div></div>'
    )


def status_item(label: str, value: Any, css: str = "") -> str:
    return (
        '<div class="status-item">'
        f'<div class="status-key">{html.escape(label)}</div>'
        f'<div class="status-val {css}">{html.escape(str(value))}</div></div>'
    )


def render_status() -> None:
    livez = get_json("/livez")
    trading = get_json("/health/trading")
    mode_api = get_json("/trading/status")
    broker = (trading or {}).get("broker") or {}
    recon = (trading or {}).get("reconciliation") or {}
    blockers = list((trading or {}).get("blockers") or [])
    stats = event_ring().stats()
    feed_fresh = bool(stats.get("last_event")) and clock.time() - float(stats["last_event"]) < 35
    process_up = livez is not None or feed_fresh
    engine_loaded = bool((livez or {}).get("bot_loaded"))
    mode = str((mode_api or {}).get("execution_mode") or "UNKNOWN").upper()
    shadow = mode != "LIVE" or "not_live_mode" in blockers
    orders_armed = bool((trading or {}).get("live_orders_armed"))
    broker_ready = bool(broker.get("ready")) if broker else False
    reconciled = bool(recon.get("completed")) if recon else False
    balance = broker.get("balance")

    st.markdown(
        '<div class="cards">'
        + card("Process", "UP" if process_up else "DOWN", "good-text" if process_up else "bad-text")
        + card("Engine", "LOADED" if engine_loaded else "STARTING", "good-text" if engine_loaded else "warn-text")
        + card("Mode", mode, "warn-text" if shadow else "good-text")
        + card("Broker", "READY" if broker_ready else "UNKNOWN", "good-text" if broker_ready else "warn-text")
        + card("Balance", f"₹{float(balance):,.2f}" if isinstance(balance, (int, float)) else "—")
        + "</div>",
        unsafe_allow_html=True,
    )

    if shadow and broker_ready and reconciled:
        alert = '<div class="alert ok"><b>SHADOW mode — safe monitoring</b><br>Broker and reconciliation are healthy; real orders are intentionally disabled.</div>'
    elif orders_armed and not blockers:
        alert = '<div class="alert ok"><b>LIVE execution armed</b><br>No operational blocker reported.</div>'
    elif blockers:
        meaningful = [item for item in blockers if item != "not_live_mode"]
        if meaningful:
            alert = '<div class="alert warn"><b>Execution blocked</b><br>' + " · ".join(html.escape(str(item)) for item in meaningful) + "</div>"
        else:
            alert = '<div class="alert ok"><b>Real orders disabled by configuration</b><br>The bot remains available for observation and paper evaluation.</div>'
    elif not process_up:
        alert = '<div class="alert bad"><b>Bot unavailable</b><br>No API response and no recent journal event.</div>'
    else:
        alert = '<div class="alert warn"><b>Startup or status transition</b><br>Waiting for a complete runtime snapshot.</div>'
    st.markdown(alert, unsafe_allow_html=True)

    age = f"{int(clock.time() - stats['last_event'])}s" if stats.get("last_event") else "waiting"
    state = updater_state()
    st.markdown(
        '<div class="card"><div class="label" style="margin-bottom:7px">Runtime details</div>'
        '<div class="status-grid">'
        + status_item("Orders", "ARMED" if orders_armed else "OFF", "good-text" if orders_armed else "warn-text")
        + status_item("Reconciled", "YES" if reconciled else "NO", "good-text" if reconciled else "warn-text")
        + status_item("Event follower", "CONNECTED" if stats.get("connected") else "RECONNECTING", "good-text" if stats.get("connected") else "warn-text")
        + status_item("Last event", age)
        + status_item("Buffer", f"{int(stats.get('size') or 0):,}/{int(stats.get('capacity') or 0):,}")
        + status_item("Feed errors", stats.get("last_error") or "NONE", "good-text" if not stats.get("last_error") else "warn-text")
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="card"><div class="label" style="margin-bottom:6px">Deployment</div>'
        f'<div class="deploy"><span>Platform</span><span>{html.escape(os.getenv("DEPLOYMENT_PLATFORM", "aws_lightsail"))}</span></div>'
        f'<div class="deploy"><span>Running</span><span>{html.escape(git_commit("HEAD"))}</span></div>'
        f'<div class="deploy"><span>Remote main</span><span>{html.escape(git_commit("origin/main"))}</span></div>'
        f'<div class="deploy"><span>Updater</span><span>{html.escape(str(state.get("state", "unknown")))}</span></div>'
        f'<div class="deploy"><span>Message</span><span>{html.escape(str(state.get("message", "No report")))}</span></div></div>',
        unsafe_allow_html=True,
    )
    st.session_state["diagnostics"] = {
        "livez": livez,
        "trading": trading,
        "mode": mode_api,
        "event_transport": stats,
        "updater": state,
    }


now = datetime.now(IST)
within_session = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
st.markdown(
    '<div class="hero"><div><div class="brand">⚡ Nifty Scalper Operations</div>'
    '<div class="sub">Fast read-only status · actionable events · bounded log exports</div></div>'
    '<div class="pills">'
    f'<span class="pill {"good" if within_session else "warn"}">{"SESSION OPEN" if within_session else "OUTSIDE SESSION"}</span>'
    f'<span class="pill">{now:%d %b %Y · %I:%M %p IST}</span><span class="pill">READ ONLY</span></div></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    controls = st.columns([.8, 1, 1.1, 2.2, .9, .8], gap="small")
    live_refresh = controls[0].toggle("Live", value=True)
    refresh_seconds = controls[1].selectbox("Refresh", [2, 3, 5, 10], index=1, format_func=lambda value: f"{value}s")
    event_type = controls[2].selectbox("Event", EVENT_TYPES)
    query = controls[3].text_input("Contains", placeholder="symbol, order, blocker, rejection…")
    row_limit = controls[4].selectbox("Rows", [50, 100, 200, 300], index=1)
    trades_view = controls[5].toggle("Trades", value=False)

feed_column, status_column = st.columns([3.45, 1.15], gap="medium")
events = event_ring()


def render_live_feed() -> None:
    snapshot = events.snapshot()
    rows = filter_events(snapshot, event_type, query)
    if trades_view:
        rows = trades_only(rows)
    rows = rows[-row_limit:]
    stats = events.stats()
    pnl_total, closed, wins = realized_pnl(snapshot)
    win_rate = round(100 * wins / closed) if closed else 0
    st.markdown(
        '<div class="cards">'
        + card("Visible", len(rows))
        + card("Trades", sum(row.get("type") == "TRADE" for row in rows))
        + card("Signals", sum(row.get("type") == "SIGNAL" for row in rows))
        + card("Errors", sum(row.get("type") == "ERROR" for row in rows), "bad-text" if any(row.get("type") == "ERROR" for row in rows) else "")
        + card("Realized P&L", f"₹{pnl_total:,.2f} · {win_rate}%", "good-text" if pnl_total >= 0 else "bad-text")
        + "</div>" + render_feed(rows)
        + '<div class="foot">'
        f'<span>Follower: {"connected" if stats.get("connected") else "reconnecting"}</span>'
        f'<span>Buffer: {int(stats.get("size") or 0):,}</span>'
        f'<span>Updated: {datetime.now(IST):%I:%M:%S %p}</span></div>',
        unsafe_allow_html=True,
    )


with feed_column:
    st.markdown('<div class="section">Actionable event stream</div>', unsafe_allow_html=True)
    if live_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{refresh_seconds}s")
        def live_fragment() -> None:
            render_live_feed()
        live_fragment()
    else:
        render_live_feed()

    current_rows = filter_events(events.snapshot(), event_type, query)
    if trades_view:
        current_rows = trades_only(current_rows)
    st.download_button(
        "Download current filtered buffer",
        csv_bytes(current_rows),
        file_name=f"niftybot-live-buffer-{datetime.now(IST):%Y%m%d-%H%M}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with status_column:
    st.markdown('<div class="section">System and trading status</div>', unsafe_allow_html=True)
    if hasattr(st, "fragment"):
        @st.fragment(run_every="4s")
        def status_fragment() -> None:
            render_status()
        status_fragment()
    else:
        render_status()

with st.expander("History and downloads", expanded=False):
    export_top = st.columns([1.2, 1.05, 1.05, 1.2], gap="small")
    preset = export_top[0].selectbox("Window", ["Market session", "Full day", "Last 2 hours", "Custom"])
    selected_date = export_top[1].date_input("Date", date.today(), max_value=date.today())
    source = export_top[2].selectbox("Export", ["Actionable events CSV", "Full service log TXT"])
    history_type = export_top[3].selectbox("Event type", EVENT_TYPES, disabled=source.startswith("Full"))

    current = datetime.now(IST)
    if preset == "Market session":
        start_at, end_at = time(9, 0), time(15, 45)
    elif preset == "Full day":
        start_at, end_at = time(0, 0), time(23, 59, 59)
    elif preset == "Last 2 hours":
        selected_date = current.date()
        start_at = (current - timedelta(hours=2)).time().replace(microsecond=0)
        end_at = current.time().replace(microsecond=0)
    else:
        custom = st.columns(2)
        start_at = custom[0].time_input("From · IST", time(9, 15), step=60)
        end_at = custom[1].time_input("To · IST", time(15, 30), step=60)

    options = st.columns([2.2, 1, 1], gap="small")
    history_query = options[0].text_input("Contains", placeholder="optional exact filter", key="export_contains")
    history_trades = options[1].toggle("Trades only", value=False, disabled=source.startswith("Full"))
    preview_limit = options[2].selectbox("Preview rows", [100, 250, 500], index=1)

    generate, clear = st.columns(2)
    if clear.button("Clear export", use_container_width=True):
        st.session_state.pop("generated_export", None)
        st.rerun()

    if generate.button("Generate download", type="primary", use_container_width=True):
        if start_at >= end_at:
            st.error("From time must be earlier than To time.")
        else:
            with st.spinner("Reading a bounded journal window…"):
                if source.startswith("Actionable"):
                    rows, result = read_actionable_events(SERVICE, selected_date, start_at, end_at)
                    rows = filter_events(rows, history_type, history_query)
                    if history_trades:
                        rows = trades_only(rows)
                    st.session_state["generated_export"] = {
                        "kind": "events",
                        "data": csv_bytes(rows),
                        "filename": f"niftybot-events-{selected_date}-{start_at:%H%M}-{end_at:%H%M}.csv",
                        "mime": "text/csv",
                        "count": len(rows),
                        "truncated": result.truncated,
                        "preview": rows[-preview_limit:],
                        "error": result.error,
                    }
                else:
                    result = read_raw_logs(SERVICE, selected_date, start_at, end_at, history_query)
                    preview = result.data.decode("utf-8", errors="replace").splitlines()[-preview_limit:]
                    st.session_state["generated_export"] = {
                        "kind": "raw",
                        "data": result.data,
                        "filename": f"niftybot-full-log-{selected_date}-{start_at:%H%M}-{end_at:%H%M}.txt",
                        "mime": "text/plain",
                        "count": result.count,
                        "truncated": result.truncated,
                        "preview": preview,
                        "error": result.error,
                    }

    export = st.session_state.get("generated_export")
    if export:
        if export.get("error"):
            st.error(export["error"])
        else:
            message = f"Prepared {int(export.get('count') or 0):,} rows."
            if export.get("truncated"):
                message += " Safety limit reached; the newest 24 MB were retained."
            st.success(message)
            if export.get("kind") == "events":
                st.dataframe(
                    export.get("preview") or [],
                    column_order=["timestamp_ist", "type", "message"],
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )
            else:
                st.code("\n".join(export.get("preview") or []), language="text", line_numbers=False)
            st.download_button(
                f"Download · {export['filename']}",
                export["data"],
                file_name=export["filename"],
                mime=export["mime"],
                type="primary",
                use_container_width=True,
            )
    else:
        st.info("Exports run only when requested. Live refresh never scans historical journal data.")

with st.expander("Technical diagnostics", expanded=False):
    st.json(st.session_state.get("diagnostics", {}))
