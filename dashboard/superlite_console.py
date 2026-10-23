"""Purpose: Render the low-memory read-only Streamlit operations console.

Responsibilities:
- status_snapshot, service_health_snapshot and recent_events fetch bounded data.
- render shows readiness, engine health, P&L summaries and event feed.
- Display labels distinguish missing evidence from confirmed broker state.

Boundaries: Console does not run strategy, orders or release activation.
Safety: A responsive UI is not proof that the live trading engine is ready.
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time
from zoneinfo import ZoneInfo

import streamlit as st

from dashboard.superlite_events import csv_bytes, deduplicate_events, filter_events, parse_event
from nifty_scalper_bot.admin_install_proof_display import install_proof_display

IST = ZoneInfo("Asia/Kolkata")
ADMIN_API = os.getenv("BOT_ADMIN_API_URL", "http://127.0.0.1:8081").rstrip("/")
_configured_admin = os.getenv("BOT_ADMIN_PUBLIC_URL", "").strip()
# Older systemd units override the environment file with a retired static IP.
# Recover that exact legacy default without changing an operator-set custom URL.
if (
    not _configured_admin
    or urllib.parse.urlsplit(_configured_admin).hostname == "15.206.3.6"
):
    ADMIN_PUBLIC = "http://" + os.getenv("BOT_PUBLIC_IP", "65.0.255.84") + ":8081/admin"
else:
    ADMIN_PUBLIC = _configured_admin
SERVICE = os.getenv("BOT_SERVICE_NAME", "niftybot")
LOW_MEMORY_MODE = os.getenv("BOT_CONSOLE_LOW_MEMORY_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
STATUS_TTL = max(3, int(os.getenv("BOT_CONSOLE_STATUS_TTL_SECONDS", "10" if LOW_MEMORY_MODE else "6")))
EVENT_TTL = max(5, int(os.getenv("BOT_CONSOLE_EVENT_TTL_SECONDS", "15" if LOW_MEMORY_MODE else "8")))
MAX_JOURNAL_LINES = max(100, min(int(os.getenv("BOT_CONSOLE_MAX_JOURNAL_LINES", "300" if LOW_MEMORY_MODE else "650")), 1000))
EVENT_TYPES = ["ALL", "TRADE", "SIGNAL", "RISK", "ERROR", "WARNING", "SYSTEM"]
PNL = re.compile(r"\bpnl=(-?\d+(?:\.\d+)?)", re.IGNORECASE)

st.set_page_config(page_title="Nifty Scalper Review", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
:root{--bg:#070b11;--panel:#0d151f;--line:#223249;--text:#e7edf5;--muted:#8292a7;--green:#39d98a;--amber:#f4c45d;--red:#ff6475;--blue:#67a9ff}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)}
header,[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}.block-container{max-width:none!important;padding:.55rem .85rem 1rem!important}
.hero{background:#0d1d2c;border:1px solid #29445e;border-radius:12px;padding:11px 14px;margin-bottom:8px;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.hero b{font-size:1.15rem}.muted{color:var(--muted);font-size:.72rem}.cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin:7px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:9px}.label{font-size:.61rem;color:var(--muted);text-transform:uppercase}.value{font-size:.95rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.feed{height:470px;overflow:auto;background:#05090e;border:1px solid var(--line);border-radius:10px}.row{display:grid;grid-template-columns:140px 65px minmax(0,1fr);gap:8px;padding:5px 8px;border-bottom:1px solid #152233;font:11.5px/1.4 ui-monospace,Consolas,monospace}.ts{color:#7f91a6}.msg{word-break:break-word}.ERROR{color:var(--red)}.WARNING{color:var(--amber)}.TRADE{color:#55dfd4}.SIGNAL{color:var(--blue)}.RISK{color:#c39aff}.SYSTEM{color:#aebdcc}@media(max-width:750px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}.row{grid-template-columns:100px 55px minmax(0,1fr);font-size:9.5px}}
</style>
""", unsafe_allow_html=True)


def _get_json(path: str) -> dict:
    try:
        with urllib.request.urlopen(ADMIN_API + path, timeout=1.4) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else {}
    except TimeoutError:
        return {"engine_http_status": "ADMIN API UNREACHABLE"}
    except (OSError, ValueError, urllib.error.URLError):
        return {"engine_http_status": "ADMIN API UNREACHABLE"}


def _post(path: str) -> tuple[bool, str]:
    """Read the operator service's actual verdict, not its HTTP redirect."""
    request = urllib.request.Request(
        ADMIN_API + path,
        data=b"",
        method="POST",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except (OSError, ValueError):
            return False, f"Control rejected (HTTP {exc.code})"
        return False, str(
            payload.get("message") or f"Control rejected (HTTP {exc.code})"
        )
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return False, "Admin control service unreachable or invalid response"
    if not isinstance(payload, dict):
        return False, "Admin returned invalid control response"
    ok = payload.get("ok") is True and payload.get("state") == "queued"
    return ok, str(payload.get("message") or "Unknown request status")


@st.cache_data(ttl=STATUS_TTL, show_spinner=False)
def status_snapshot() -> dict:
    return _get_json("/admin/api/status")


@st.cache_data(ttl=STATUS_TTL, show_spinner=False)
def service_health_snapshot() -> dict:
    """Read only independent host-local HTTP liveness, not trading readiness."""
    return _get_json("/admin/api/service-health")


@st.cache_data(ttl=EVENT_TTL, show_spinner=False)
def recent_events(lines: int = MAX_JOURNAL_LINES) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["journalctl", "-u", SERVICE, "-n", str(max(100, min(lines, MAX_JOURNAL_LINES))), "--no-pager", "-o", "cat"],
            capture_output=True,
            text=True,
            timeout=3 if LOW_MEMORY_MODE else 5,
            check=False,
        )
        raw = result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        raw = ""
    return deduplicate_events(event for line in raw.splitlines() if (event := parse_event(line)))


def card(label: str, value: object, css: str = "") -> str:
    return f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value {css}">{html.escape(str(value))}</div></div>'


def broker_display_label(value: object) -> str:
    if value == "authenticated":
        return "YES/READY"
    if value == "invalid":
        return "NO/FAILED"
    return "UNKNOWN"


def reconciliation_display_label(value: object, reconciliation: dict) -> str:
    if value is True:
        return "YES/READY"
    if bool(reconciliation.get("failed")):
        return "NO/FAILED"
    return "UNKNOWN"


def trading_ready_status(status: dict) -> bool:
    mode = str(status.get("mode") or status.get("execution_mode") or "").upper()
    blockers = [str(value) for value in status.get("blockers", []) if str(value).strip()]
    return (
        mode == "LIVE"
        and bool(status.get("live_orders_armed"))
        and status.get("broker_authenticated") == "authenticated"
        and status.get("reconciled") is True
        and not blockers
    )


def engine_display_status(status: dict) -> str:
    if status.get("engine_http_responsive") is False:
        return status.get("engine_http_status") or "ENGINE HTTP UNRESPONSIVE"
    if not status.get("engine_loaded"):
        return status.get("engine_http_status") or "BOT NOT LOADED"
    if trading_ready_status(status):
        return "ENGINE UP, TRADING READY"
    if status.get("operational_ready"):
        return "ENGINE UP, OPERATIONAL"
    return "ENGINE UP, TRADING BLOCKED"


def hook_count_label(proof_display: dict) -> str:
    counts = proof_display.get("hook_counts") if isinstance(proof_display, dict) else {}
    if not isinstance(counts, dict) or not counts:
        return "—"
    return f"core:{counts.get('core_app', '—')} data:{counts.get('datahub', '—')}"


def missing_hardening_label(proof_display: dict) -> str:
    missing = proof_display.get("missing") if isinstance(proof_display, dict) else []
    if not missing:
        return "NONE"
    return ", ".join(str(item) for item in missing[:3]) + ("…" if len(missing) > 3 else "")


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
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass
    return round(sum(values), 2), len(values)


now = datetime.now(IST)
within_session = now.weekday() < 5 and time(9, 15) <= now.time() <= time(15, 30)
st.markdown(
    f'<div class="hero"><div><b>⚡ Nifty Scalper Review</b><div class="muted">Low-memory mode {"ON" if LOW_MEMORY_MODE else "OFF"} · bounded journal reads · no background follower</div></div><div><b>{"SESSION OPEN" if within_session else "OUTSIDE SESSION"}</b><div class="muted">{now:%d %b %Y · %I:%M %p IST}</div></div></div>',
    unsafe_allow_html=True,
)
refresh_choices = [20, 30, 60, 120] if LOW_MEMORY_MODE else [8, 10, 15, 30]
row_choices = [25, 50, 100, 150] if LOW_MEMORY_MODE else [50, 100, 200, 300]
controls = st.columns([1, 1, 1.1, 2.2, .9, .9, 1.1], gap="small")
auto_refresh = controls[0].toggle("Auto-refresh", value=False if LOW_MEMORY_MODE else within_session)
refresh_seconds = controls[1].selectbox("Every", refresh_choices, index=1, format_func=lambda value: f"{value}s")
event_type = controls[2].selectbox("Event", EVENT_TYPES)
query = controls[3].text_input("Contains", placeholder="symbol, blocker, order…")
row_limit = controls[4].selectbox("Rows", row_choices, index=1)
trade_only = controls[5].toggle("Trade events", value=False)
controls[6].link_button("Open admin", ADMIN_PUBLIC, width="stretch")

ops = st.columns([1.1, 1.2, 1.1, 2.3], gap="small")
if ops[0].button("Queue engine restart", type="secondary", width="stretch"):
    st.cache_data.clear()
    accepted, message = _post("/admin/restart")
    (st.info if accepted else st.error)(message)
if ops[1].button("Check GitHub update", type="primary", width="stretch"):
    st.cache_data.clear()
    accepted, message = _post("/admin/update")
    (st.info if accepted else st.error)(message)
ops[2].link_button("Hard admin", ADMIN_PUBLIC, width="stretch")
ops[3].caption(
    "Controls request the validated deployment service. "
    "Updates and restarts can interrupt LIVE position and stop management. "
    "A queued request is not a completed deployment."
)


def render() -> None:
    status = status_snapshot()
    services = service_health_snapshot()
    rows_all = recent_events()
    rows = filter_events(rows_all, event_type, query)
    if trade_only:
        rows = [row for row in rows if row.get("type") == "TRADE"]
    rows = rows[-row_limit:]
    broker = status.get("broker") or {}
    recon = status.get("reconciliation") or {}
    selected = status.get("selected") or {}
    memory = status.get("host_memory") or {}
    proof = status.get("install_proof") or {}
    proof_status = status.get("install_proof_display") or install_proof_display(proof)
    pnl_total, closed = pnl_summary(rows_all)
    primary = status.get("primary_blocker") or next((b for b in status.get("blockers", []) if b), "—")
    engine_status = engine_display_status(status)
    broker_state = status.get("broker_authenticated")
    broker_label = broker_display_label(broker_state)
    recon_state = status.get("reconciled")
    recon_label = reconciliation_display_label(recon_state, recon)
    mem_pct = memory.get("mem_used_pct")
    mem_css = "bad" if isinstance(mem_pct, (int, float)) and mem_pct >= 90 else "warn" if isinstance(mem_pct, (int, float)) and mem_pct >= 75 else "ok"
    hardening_css = str(proof_status.get("css") or "warn")
    st.markdown(
        '<div class="cards">'
        + card("Engine", engine_status, "ok" if status.get("operational_ready") else "warn")
        + card("Primary blocker", primary, "warn" if primary != "—" else "ok")
        + card("Hardening", proof_status.get("label") or "UNKNOWN", hardening_css)
        + card("Broker", broker_label, "ok" if broker_label.startswith("YES") else "warn")
        + card("Reconciled", recon_label, "ok" if recon_label.startswith("YES") else "warn")
        + '</div><div class="cards">'
        + card("Hook counts", hook_count_label(proof_status), hardening_css)
        + card("Missing hardening", missing_hardening_label(proof_status), "ok" if not proof_status.get("missing") else "bad")
        + card("Memory", f"{mem_pct if mem_pct is not None else '—'}%", mem_css)
        + card("Error events", sum(row.get("type") == "ERROR" for row in rows), "bad" if any(row.get("type") == "ERROR" for row in rows) else "")
        + card("Log realised P&L", f"₹{pnl_total:,.2f} · {closed} closed", "ok" if pnl_total >= 0 else "bad")
        + '</div><div class="cards">'
        + card("Visible events", len(rows))
        + card("Trade events", sum(row.get("type") == "TRADE" for row in rows))
        + card("Signal events", sum(row.get("type") == "SIGNAL" for row in rows))
        + card("Engine SHA", status.get("running") or "—")
        + card(
            "Cached main",
            status.get("remote") or "—",
            "warn" if status.get("stale") else "",
        )
        + '</div><div class="cards">'
        + card("ATM", selected.get("atm") or "—")
        + card("Selected CE", selected.get("ce") or "—")
        + card("Selected PE", selected.get("pe") or "—")
        + card("Mode", status.get("mode") or status.get("execution_mode") or "—")
        + card("Selected source", status.get("selected_source") or "—")
        + card(
            "Updater",
            (status.get("updater") or {}).get("state") or "unknown",
            "warn" if status.get("stale") else "",
        )
        + "</div>" + feed_html(rows),
        unsafe_allow_html=True,
    )
    st.download_button("Download current filtered events", csv_bytes(rows), file_name=f"niftybot-events-{datetime.now(IST):%Y%m%d-%H%M}.csv", mime="text/csv", width="stretch")
    with st.expander("Hardening install proof", expanded=False):
        st.json({"display": proof_status, "install_proof": proof})
    with st.expander("Service liveness (host-local)", expanded=False):
        if services:
            st.json(services)
            st.caption(
                "Local HTTP liveness does not verify public firewall reachability "
                "or broker/trading readiness."
            )
        else:
            st.warning("Admin service probe unavailable; liveness is unknown.")
    with st.expander("Technical status", expanded=False):
        st.json(status)


if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_seconds}s")
    def live_fragment() -> None:
        render()
    live_fragment()
else:
    render()
