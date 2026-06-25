"""Full-width, low-overhead trading operations console."""
from __future__ import annotations

import csv
import html
import io
import json
import os
import re
import subprocess
import time as clock
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from event_buffer import EventRing, deduplicate_events, parse_event

st.set_page_config(
    page_title="Nifty Scalper Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root{
  --bg:#070b11;--panel:#0c131d;--panel2:#101a26;--line:#213247;
  --text:#e6edf5;--muted:#8292a7;--green:#35d07f;--amber:#f5c451;
  --red:#ff5d6c;--cyan:#55dfd4;--blue:#65a9ff;--purple:#b58cff;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important}
header[data-testid="stHeader"]{display:none!important}
[data-testid="stSidebar"],[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{display:none!important}
[data-testid="stAppViewBlockContainer"],.block-container{
  max-width:none!important;width:100%!important;padding:.65rem 1rem 1.25rem!important;
}
[data-testid="stToolbar"]{display:none!important}
.terminal-head{
  display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:14px;
  background:linear-gradient(135deg,#0b1a29,#10283a);
  border:1px solid #29445e;border-radius:14px;padding:13px 16px;margin:0 0 10px;
  box-shadow:0 8px 30px rgba(0,0,0,.18);
}
.brand{font-size:1.35rem;font-weight:780;letter-spacing:.01em;color:var(--text)}
.tagline{font-size:.76rem;color:#91a3b7;margin-top:2px}
.head-right{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.pill{border:1px solid #35516d;border-radius:999px;padding:5px 10px;font-size:.72rem;
  font-weight:750;color:#bed0e2;background:#0b1723}
.pill.open{color:#66e3a1;border-color:#286c4a;background:#0e241b}
.pill.closed{color:#b9c3cf}
.section-title{font-size:.96rem;font-weight:730;color:var(--text);margin:2px 0 7px}
.control-shell{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:8px 10px 2px;margin-bottom:10px}
.status-stack{display:grid;gap:9px}
.status-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:11px 12px}
.card-title{font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);margin-bottom:8px}
.state-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.state-item{background:#09111a;border:1px solid #1d2b3a;border-radius:8px;padding:8px}
.state-label{font-size:.66rem;text-transform:uppercase;color:#74869a;letter-spacing:.06em}
.state-value{font-size:.95rem;font-weight:760;color:var(--text);margin-top:2px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.good{color:var(--green)!important}.bad-text{color:var(--red)!important}
.warn-text{color:var(--amber)!important}.info-text{color:var(--blue)!important}
.alert{border-radius:9px;padding:9px 10px;font-size:.76rem;line-height:1.38}
.alert.ok{border-left:4px solid var(--green);background:#10271e;color:#c9f5dc}
.alert.warn{border-left:4px solid var(--amber);background:#282313;color:#f5df9f}
.alert.bad{border-left:4px solid var(--red);background:#2b141a;color:#ffc3ca}
.kpi-row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-bottom:8px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 10px}
.kpi-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.kpi-value{font-size:1.02rem;font-weight:780;color:var(--text);margin-top:2px}
.feed-shell{background:#05090e;border:1px solid var(--line);border-radius:12px;overflow:hidden}
.feed-head{display:grid;grid-template-columns:146px 75px 1fr;gap:8px;padding:7px 10px;
  background:#0b141e;border-bottom:1px solid var(--line);font-size:.62rem;text-transform:uppercase;
  letter-spacing:.07em;color:#8093a8;position:sticky;top:0;z-index:2}
.feed{height:570px;overflow-y:auto;overscroll-behavior:contain}
.ev{display:grid;grid-template-columns:146px 75px minmax(0,1fr);gap:8px;padding:6px 10px;
  border-bottom:1px solid #142130;font:12.6px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
.ev:hover{background:#0a121b}.ts{color:#8093a8}.msg{color:#d8e3ef;word-break:break-word}
.badge{font-weight:800;font-size:11px}.ERROR{color:var(--red)}.WARNING{color:var(--amber)}
.TRADE{color:var(--cyan)}.SIGNAL{color:var(--blue)}.RISK{color:var(--purple)}.SYSTEM{color:#aebdcc}
.feed-empty{padding:22px;color:var(--muted);font:12px ui-monospace,monospace}
.feed-foot{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;padding:7px 10px;
  border-top:1px solid var(--line);background:#09111a;color:#718399;font-size:.66rem}
.deploy-row{display:flex;justify-content:space-between;gap:8px;border-bottom:1px solid #172536;
  padding:5px 0;font-size:.72rem}.deploy-row:last-child{border-bottom:0}
.deploy-key{color:var(--muted)}.deploy-value{color:#c9d5e2;font-family:ui-monospace,monospace}
[data-testid="stWidgetLabel"] p{font-size:.7rem!important;color:#8fa1b5!important}
[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stDateInput"] input{
  background:#09111a!important;border-color:#26384b!important}
div[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:12px!important;background:var(--panel)}
.stButton>button,.stDownloadButton>button{border-radius:8px!important;font-weight:720!important}
@media(max-width:1100px){
  .kpi-row{grid-template-columns:repeat(3,minmax(0,1fr))}
  .feed,.feed-shell{min-width:0}.feed{height:520px}
}
@media(max-width:760px){
  [data-testid="stAppViewBlockContainer"],.block-container{padding:.45rem .45rem 1rem!important}
  .terminal-head{grid-template-columns:1fr;padding:11px 12px}.head-right{justify-content:flex-start}
  .brand{font-size:1.1rem}.kpi-row{grid-template-columns:repeat(2,minmax(0,1fr))}
  .feed-head,.ev{grid-template-columns:112px 58px minmax(0,1fr)}
  .ev{font-size:9.8px;padding:6px}.feed-head{padding:6px}.feed{height:470px}
}
</style>
""",
    unsafe_allow_html=True,
)

IST = ZoneInfo("Asia/Kolkata")
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
API = os.getenv("BOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
APP = Path(os.getenv("BOT_APP_DIR", "/home/ubuntu/nifty_scalper_bot"))
UPDATE_FILE = APP / "data" / "auto_update_status.json"


@st.cache_resource
def http_session() -> requests.Session:
    return requests.Session()


@st.cache_resource
def event_ring() -> EventRing:
    return EventRing(SERVICE, max_events=3000)


@st.cache_data(ttl=2.0, show_spinner=False)
def get_json(path: str) -> dict[str, Any] | None:
    # Cached briefly: render_status_rail calls /livez, /readyz and /health/trading
    # every rerun. At a 2s auto-refresh with a 1.8s timeout each, an unreachable bot
    # API would block the page for up to ~5s per cycle (the "latency/hang"). Caching
    # for one refresh cycle keeps the terminal responsive regardless of bot-API state;
    # the event feed (journald) is independent and always live.
    try:
        response = http_session().get(API + path, timeout=1.2)
        value = response.json()
        if not isinstance(value, dict):
            return None
        value = dict(value)
        value['_http_status'] = response.status_code
        return value
    except (requests.RequestException, ValueError):
        return None


def filter_events(
    rows: list[dict[str, str]],
    event_type: str,
    query: str,
) -> list[dict[str, str]]:
    if event_type != "ALL":
        rows = [row for row in rows if row["type"] == event_type]
    needle = query.strip().lower()
    if needle:
        rows = [row for row in rows if needle in row["message"].lower()]
    return rows


# Markers for the "Trades only" view: entry, fill, exit, P&L, rejections — the
# events that describe an actual trade lifecycle (mirrors the dashboard's
# /admin/trades.json idea so trades can be reviewed without scrolling the feed).
_TRADE_MARKERS = (
    "ORDER_SENT", "Sending Order", "FILLED", "average_price", "EXIT",
    "TRADE_ATTEMPT", "ORDER_REJECTED", "ORDER_BROKER_CONFIG_ERROR",
    "BRACKET_EXIT", "EXIT_RECONCILED_FLAT", "EXIT_ESCALAT",
    "EXIT_ESCALATION_MARKET_EXIT", "pnl", "SIGNAL_EXECUTION_RESULT",
)


def filter_trades_only(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if any(m.lower() in r["message"].lower() for m in _TRADE_MARKERS)]


_PNL_RE = re.compile(r"\bpnl=(-?\d+(?:\.\d+)?)")


def realized_pnl_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Aggregate realized P&L from BRACKET_CLOSED events in the given rows.

    Each closed bracket logs `pnl=<rupees>`; sum them for the day's realized P&L,
    with win/loss counts. Returns zeros when no closes are present yet.
    """
    total = 0.0
    wins = losses = trades = 0
    best = worst = None
    for row in rows:
        msg = row.get("message", "")
        if "BRACKET_CLOSED" not in msg:
            continue
        m = _PNL_RE.search(msg)
        if not m:
            continue
        try:
            pnl = float(m.group(1))
        except ValueError:
            continue
        total += pnl
        trades += 1
        if pnl >= 0:
            wins += 1
        else:
            losses += 1
        best = pnl if best is None else max(best, pnl)
        worst = pnl if worst is None else min(worst, pnl)
    return {
        "total": round(total, 2),
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "best": best,
        "worst": worst,
    }


def render_feed(rows: list[dict[str, str]]) -> str:
    if not rows:
        body = '<div class="feed-empty">Waiting for actionable events…</div>'
    else:
        values: list[str] = []
        for row in reversed(rows):
            values.append(
                '<div class="ev">'
                f'<span class="ts">{html.escape(row["timestamp_ist"])}</span>'
                f'<span class="badge {row["type"]}">{row["type"]}</span>'
                f'<span class="msg">{html.escape(row["message"])}</span>'
                "</div>"
            )
        body = "".join(values)
    return (
        '<div class="feed-shell">'
        '<div class="feed-head"><span>Time · IST</span><span>Event</span><span>Message</span></div>'
        f'<div class="feed">{body}</div>'
        "</div>"
    )


def read_history(
    selected_date: date,
    start_at: time,
    end_at: time,
) -> tuple[list[dict[str, str]], str | None]:
    command = [
        "journalctl",
        "-u",
        SERVICE,
        "--since",
        f"{selected_date} {start_at:%H:%M:%S}",
        "--until",
        f"{selected_date} {end_at:%H:%M:%S}",
        "--no-pager",
        "-o",
        "cat",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return [], str(exc)
    if result.returncode:
        return [], result.stderr.strip() or "journal query failed"
    rows = [
        event
        for line in result.stdout.splitlines()
        if (event := parse_event(line))
    ]
    return deduplicate_events(rows), None


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    target = io.StringIO()
    writer = csv.DictWriter(
        target,
        fieldnames=["timestamp_ist", "type", "message"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return target.getvalue().encode("utf-8-sig")


def updater_state() -> dict[str, Any]:
    try:
        value = json.loads(UPDATE_FILE.read_text())
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def git_commit(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(APP), "rev-parse", "--short", ref],
            text=True,
            timeout=2,
        ).strip()
    except Exception:
        return "—"


def short_value(value: Any, fallback: str = "—") -> str:
    return fallback if value is None or value == "" else str(value)


def state_item(label: str, value: str, css_class: str = "") -> str:
    return (
        '<div class="state-item">'
        f'<div class="state-label">{html.escape(label)}</div>'
        f'<div class="state-value {css_class}">{html.escape(value)}</div>'
        "</div>"
    )


def render_status_rail() -> None:
    livez = get_json("/livez")
    readyz = get_json("/readyz")
    trading = get_json("/health/trading")
    broker = (trading or {}).get("broker") or {}
    recon = (trading or {}).get("reconciliation") or {}
    blockers = (
        (trading or {}).get("blockers")
        or (readyz or {}).get("blockers")
        or []
    )
    # Decouple from the status API: the event feed (journald) is independent of the
    # bot's HTTP endpoints. If /livez is unreachable but events arrived in the last
    # ~30s, the bot IS running — the status endpoint is just briefly unreachable
    # (e.g. threadpool busy). Show that truthfully instead of a misleading "DOWN".
    _fs = event_ring().stats()
    _feed_fresh = bool(_fs.get("last_event")) and (clock.time() - _fs["last_event"]) < 30.0
    api_up = livez is not None
    process_alive = api_up or _feed_fresh
    process_label = "UP" if api_up else ("LIVE (feed)" if _feed_fresh else "DOWN")
    process_class = "good" if process_alive else "bad-text"
    engine_loaded = bool((livez or {}).get("bot_loaded"))
    execution_ready = bool((trading or {}).get("ready"))
    orders_armed = bool((trading or {}).get("live_orders_armed"))
    trading_available = trading is not None
    broker_available = trading_available and bool(broker)
    recon_available = trading_available and bool(recon)
    broker_ready = bool(broker.get("ready")) if broker_available else None
    auth_invalid = bool(broker.get("auth_invalid")) if broker_available else None
    reconciled = bool(recon.get("completed")) if recon_available else None

    system_html = (
        '<div class="status-card"><div class="card-title">Execution state</div>'
        '<div class="state-grid">'
        + state_item("Process", process_label, process_class)
        + state_item("Engine", "LOADED" if engine_loaded else ("LIVE (feed)" if _feed_fresh and not api_up else "STARTING"), "good" if (engine_loaded or (_feed_fresh and not api_up)) else "warn-text")
        + state_item("Execution", "READY" if execution_ready else "BLOCKED", "good" if execution_ready else "bad-text")
        + state_item("Orders", "ARMED" if orders_armed else "OFF", "good" if orders_armed else "warn-text")
        + "</div></div>"
    )
    broker_html = (
        '<div class="status-card"><div class="card-title">Broker & account</div>'
        '<div class="state-grid">'
        + state_item("Broker", "UNKNOWN" if broker_ready is None else ("READY" if broker_ready else "NOT READY"), "warn-text" if broker_ready is None else ("good" if broker_ready else "bad-text"))
        + state_item("Balance", short_value(broker.get("balance")) if broker_available else "UNKNOWN")
        + state_item("Reconciled", "UNKNOWN" if reconciled is None else ("YES" if reconciled else "NO"), "warn-text" if reconciled is None else ("good" if reconciled else "warn-text"))
        + state_item("Authentication", "UNKNOWN" if auth_invalid is None else ("INVALID" if auth_invalid else "OK"), "warn-text" if auth_invalid is None else ("bad-text" if auth_invalid else "good"))
        + "</div></div>"
    )
    if blockers:
        alert = (
            '<div class="alert warn"><b>Blocking conditions</b><br>'
            + "<br>".join(html.escape(str(item)) for item in blockers)
            + "</div>"
        )
    elif api_up:
        alert = '<div class="alert ok"><b>Operational checks passed</b><br>No active blocker reported.</div>'
    elif _feed_fresh:
        alert = (
            '<div class="alert warn"><b>Status API unreachable — bot is live</b><br>'
            "Events are still streaming, so the engine is running; only the HTTP status "
            "endpoint is briefly unreachable. Execution/broker fields below may be stale.</div>"
        )
    else:
        alert = '<div class="alert bad"><b>Bot API unreachable & no recent events</b><br>Check the trading service.</div>'

    feed_stats = event_ring().stats()
    age = (
        f"{int(clock.time() - feed_stats['last_event'])}s ago"
        if feed_stats["last_event"]
        else "Waiting"
    )
    feed_html = (
        '<div class="status-card"><div class="card-title">Event transport</div>'
        '<div class="state-grid">'
        + state_item("Follower", "CONNECTED" if feed_stats["connected"] else "RECONNECTING",
                     "good" if feed_stats["connected"] else "warn-text")
        + state_item("Last event", age)
        + state_item("Buffer", f"{feed_stats['size']:,}")
        + state_item("Restarts", str(feed_stats["restarts"]))
        + "</div></div>"
    )

    state = updater_state()
    deploy_html = (
        '<div class="status-card"><div class="card-title">Deployment</div>'
        f'<div class="deploy-row"><span class="deploy-key">Platform</span><span class="deploy-value">{html.escape(os.getenv("DEPLOYMENT_PLATFORM", "aws_lightsail"))}</span></div>'
        f'<div class="deploy-row"><span class="deploy-key">Running</span><span class="deploy-value">{html.escape(git_commit("HEAD"))}</span></div>'
        f'<div class="deploy-row"><span class="deploy-key">Remote main</span><span class="deploy-value">{html.escape(git_commit("origin/main"))}</span></div>'
        f'<div class="deploy-row"><span class="deploy-key">Updater</span><span class="deploy-value">{html.escape(str(state.get("state", "not configured")))}</span></div>'
        f'<div class="deploy-row"><span class="deploy-key">Message</span><span class="deploy-value">{html.escape(str(state.get("message", "No updater report")))}</span></div>'
        "</div>"
    )
    st.markdown(
        '<div class="status-stack">'
        + system_html
        + broker_html
        + alert
        + feed_html
        + deploy_html
        + "</div>",
        unsafe_allow_html=True,
    )
    st.session_state["diagnostics"] = {
        "livez": livez,
        "readyz": readyz,
        "trading": trading,
        "event_transport": feed_stats,
        "updater": state,
    }


now = datetime.now(IST)
market_open = (
    now.weekday() < 5
    and time(9, 15) <= now.time() <= time(15, 30)
)
market_label = "MARKET OPEN" if market_open else "MARKET CLOSED"
market_class = "open" if market_open else "closed"

st.markdown(
    '<div class="terminal-head">'
    '<div><div class="brand">⚡ Nifty Scalper Terminal</div>'
    '<div class="tagline">Live execution state · actionable events · exact-window export</div></div>'
    f'<div class="head-right"><span class="pill {market_class}">{market_label}</span>'
    f'<span class="pill">{now:%d %b %Y}</span>'
    '<span class="pill">READ-ONLY</span></div></div>',
    unsafe_allow_html=True,
)

# Theme palettes — overrides the :root variables defined in the base stylesheet so
# the whole terminal recolors instantly. "Default dark" keeps the original look.
_THEMES = {
    "Default dark": {},
    "Midnight (high contrast)": {
        "--bg": "#02050a", "--panel": "#0a121d", "--panel2": "#0e1a28",
        "--line": "#2b4562", "--text": "#f3f8ff", "--muted": "#9fb2c8",
    },
    "Slate": {
        "--bg": "#0f1419", "--panel": "#171f29", "--panel2": "#1d2733",
        "--line": "#2e3d4f", "--text": "#eef3f9", "--muted": "#94a6ba",
    },
    "Light": {
        "--bg": "#f4f7fb", "--panel": "#ffffff", "--panel2": "#eef3f9",
        "--line": "#d4deea", "--text": "#142231", "--muted": "#5a6b80",
    },
}
_theme_choice = st.session_state.get("terminal_theme", "Default dark")
_overrides = _THEMES.get(_theme_choice) or {}
if _overrides:
    _vars = ";".join(f"{k}:{v}" for k, v in _overrides.items())
    st.markdown(f"<style>:root{{{_vars}}}</style>", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-title">Live feed controls</div>', unsafe_allow_html=True)
    with st.container(border=True):
        control_columns = st.columns([1.0, 1.05, 1.05, 1.9, 0.95, 1.15, 0.75], gap="small")
        live_refresh = control_columns[0].toggle("Live", value=True)
        refresh_seconds = control_columns[1].selectbox(
            "Refresh",
            [1, 2, 3, 5, 10],
            index=1,
            format_func=lambda value: f"{value} sec",
        )
        event_type = control_columns[2].selectbox(
            "Event type",
            ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"],
        )
        search_query = control_columns[3].text_input(
            "Search",
            placeholder="symbol, order, rejection, blocker…",
        )
        event_limit = control_columns[4].selectbox(
            "Rows",
            [50, 100, 200, 300, 500],
            index=2,
        )
        _theme_keys = list(_THEMES.keys())
        control_columns[5].selectbox(
            "Theme",
            _theme_keys,
            index=_theme_keys.index(_theme_choice) if "terminal_theme" not in st.session_state else None,
            key="terminal_theme",
        )
        if control_columns[6].button("Refresh", use_container_width=True):
            st.rerun()
    trades_only = st.toggle(
        "Trades only (entries · exits · fills · P&L)",
        value=False,
        key="trades_only_view",
        help="Show only the trade lifecycle events, hiding signal/system noise.",
    )

feed_column, rail_column = st.columns([3.35, 1.15], gap="medium")

events = event_ring()


def render_live_feed() -> None:
    snapshot = events.snapshot()
    rows = filter_events(snapshot, event_type, search_query)
    if st.session_state.get("trades_only_view"):
        rows = filter_trades_only(rows)
    rows = rows[-event_limit:]
    stats = events.stats()

    # Daily realized P&L (from BRACKET_CLOSED pnl= across the whole buffer, not just
    # the visible window) so the day's running result is visible at a glance.
    pnl = realized_pnl_summary(snapshot)
    if pnl["trades"]:
        win_rate = round(100.0 * pnl["wins"] / pnl["trades"])
        pnl_cls = "good" if pnl["total"] >= 0 else "bad-text"
        sign = "+" if pnl["total"] >= 0 else "−"
        st.markdown(
            '<div class="kpi-row" style="grid-template-columns:repeat(5,minmax(0,1fr))">'
            f'<div class="kpi"><div class="kpi-label">Realized P&amp;L (today)</div>'
            f'<div class="kpi-value {pnl_cls}">{sign}₹{abs(pnl["total"]):,.2f}</div></div>'
            f'<div class="kpi"><div class="kpi-label">Closed trades</div><div class="kpi-value">{pnl["trades"]}</div></div>'
            f'<div class="kpi"><div class="kpi-label">Win rate</div><div class="kpi-value">{win_rate}% ({pnl["wins"]}/{pnl["trades"]})</div></div>'
            f'<div class="kpi"><div class="kpi-label">Best</div><div class="kpi-value good">+₹{(pnl["best"] or 0):,.0f}</div></div>'
            f'<div class="kpi"><div class="kpi-label">Worst</div><div class="kpi-value bad-text">₹{(pnl["worst"] or 0):,.0f}</div></div>'
            "</div>",
            unsafe_allow_html=True,
        )
    values = [
        ("Visible", len(rows)),
        ("Trades", sum(row["type"] == "TRADE" for row in rows)),
        ("Signals", sum(row["type"] == "SIGNAL" for row in rows)),
        ("Risk", sum(row["type"] == "RISK" for row in rows)),
        ("Errors", sum(row["type"] == "ERROR" for row in rows)),
        (
            "Last event",
            f"{int(clock.time() - stats['last_event'])}s"
            if stats["last_event"]
            else "—",
        ),
    ]
    kpis = '<div class="kpi-row">' + "".join(
        '<div class="kpi">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        "</div>"
        for label, value in values
    ) + "</div>"
    st.markdown(kpis + render_feed(rows), unsafe_allow_html=True)
    st.markdown(
        '<div class="feed-foot">'
        f'<span>Follower: {"connected" if stats["connected"] else "reconnecting"}</span>'
        f'<span>Buffer: {stats["size"]:,} / {int(stats.get("capacity") or 3000):,}</span>'
        f'<span>Updated: {datetime.now(IST):%I:%M:%S %p IST}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


with feed_column:
    st.markdown('<div class="section-title">Actionable event stream</div>', unsafe_allow_html=True)
    if live_refresh and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{refresh_seconds}s")
        def live_feed_fragment() -> None:
            render_live_feed()

        live_feed_fragment()
    else:
        render_live_feed()

with rail_column:
    st.markdown('<div class="section-title">Operations status</div>', unsafe_allow_html=True)
    if hasattr(st, "fragment"):
        @st.fragment(run_every="3s")
        def status_rail_fragment() -> None:
            render_status_rail()

        status_rail_fragment()
    else:
        render_status_rail()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

with st.expander("History, review and CSV export", expanded=False):
    history_columns = st.columns([1.15, 1, 1, 1, 1.55], gap="small")
    selected_date = history_columns[0].date_input(
        "Date",
        date.today(),
        max_value=date.today(),
    )
    start_at = history_columns[1].time_input(
        "From",
        time(9, 15),
        step=60,
    )
    end_at = history_columns[2].time_input(
        "To",
        time(15, 30),
        step=60,
    )
    history_type = history_columns[3].selectbox(
        "Event type",
        ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"],
        key="history_type",
    )
    history_query = history_columns[4].text_input(
        "Contains",
        key="history_query",
        placeholder="optional filter",
    )
    load_column, clear_column = st.columns(2)
    load_history = load_column.button(
        "Load selected window",
        type="primary",
        use_container_width=True,
    )
    if clear_column.button("Clear loaded history", use_container_width=True):
        st.session_state.pop("history_rows", None)
        st.session_state.pop("history_error", None)

    if start_at >= end_at:
        st.error("From time must be earlier than To time.")
    elif load_history:
        with st.spinner("Reading selected journal window…"):
            (
                st.session_state["history_rows"],
                st.session_state["history_error"],
            ) = read_history(selected_date, start_at, end_at)

    history_error = st.session_state.get("history_error")
    loaded_rows = st.session_state.get("history_rows")
    if history_error:
        st.error(history_error)
    elif loaded_rows is not None:
        rows = filter_events(loaded_rows, history_type, history_query)
        if st.checkbox("Trades only (entries · exits · fills · P&L)", key="history_trades_only"):
            rows = filter_trades_only(rows)
        table_tab, preview_tab = st.tabs(["Structured table", "Event preview"])
        with table_tab:
            st.dataframe(
                rows,
                column_order=["timestamp_ist", "type", "message"],
                use_container_width=True,
                hide_index=True,
                height=340,
            )
        with preview_tab:
            st.markdown(render_feed(rows[-500:]), unsafe_allow_html=True)
        filename = (
            f"niftybot-events-{selected_date}-"
            f"{start_at:%H%M}-{end_at:%H%M}.csv"
        )
        st.download_button(
            f"⬇️ Download CSV · {len(rows):,} events",
            csv_bytes(rows),
            file_name=filename,
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    else:
        st.info("Choose a date and time window, then load history. No journal query runs until you press the button.")

with st.expander("Technical diagnostics", expanded=False):
    st.json(st.session_state.get("diagnostics", {}))
