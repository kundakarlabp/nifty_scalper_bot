from __future__ import annotations

import csv
import html
import io
import os
import re
import subprocess
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Nifty Scalper Console", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:.8rem;padding-bottom:2rem}
div[data-testid="stMetric"]{border:1px solid #29313d;border-radius:12px;padding:.65rem .8rem;background:#10151d}
div[data-testid="stMetricValue"]{font-size:1.2rem}
.logbox{height:520px;overflow-y:auto;background:#080b10;border:1px solid #29313d;border-radius:10px;padding:12px;font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
.err{color:#ff6b6b}.warn{color:#ffd166}.trade{color:#5eead4}.info{color:#cbd5e1}
@media(max-width:700px){.block-container{padding-left:.55rem;padding-right:.55rem}.logbox{height:430px;font-size:11px}}
</style>
""", unsafe_allow_html=True)

SERVICE_NAME = os.getenv("BOT_SERVICE_NAME", "niftybot")
BOT_API_URL = os.getenv("BOT_API_URL", "http://127.0.0.1:8080").rstrip("/")
STAMP_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} IST)\]")
TRADE_MARKERS = ("ORDER_SENT", "FILLED", "TRADE_ATTEMPT", "ORDER_REJECTED", "TRADE_CLOSED", "ORDER_COMPLETE", "SIGNAL_GENERATED", "EXIT", "PNL", "TARGET", "STOP_LOSS")


def api_json(path: str) -> dict[str, Any] | None:
    try:
        response = requests.get(f"{BOT_API_URL}{path}", timeout=4)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def journal_text(lines: int = 1000, since: str = "", until: str = "") -> tuple[str, str | None]:
    cmd = ["journalctl", "-u", SERVICE_NAME, "--no-pager", "-o", "cat", "-n", str(lines)]
    if since:
        cmd.extend(["--since", since])
    if until:
        cmd.extend(["--until", until])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except Exception as exc:
        return "", str(exc)
    if result.returncode != 0:
        return "", result.stderr.strip() or f"journalctl exited with {result.returncode}"
    return result.stdout, None


def classify(message: str) -> str:
    upper = message.upper()
    if "ERROR" in upper or "FAIL" in upper or "TRACEBACK" in upper or "❌" in message:
        return "ERROR"
    if "WARN" in upper or "⚠" in message:
        return "WARNING"
    if any(marker in upper for marker in TRADE_MARKERS):
        return "TRADE"
    return "INFO"


def parse_logs(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in text.splitlines():
        match = STAMP_RE.search(raw)
        if not match:
            continue
        message = raw[match.end():].strip()
        rows.append({"timestamp_ist": match.group(1), "level": classify(message), "message": message})
    return rows


def filtered(rows: list[dict[str, str]], level: str, search: str, trades_only: bool = False) -> list[dict[str, str]]:
    output = rows
    if level != "ALL":
        output = [row for row in output if row["level"] == level]
    if trades_only:
        output = [row for row in output if row["level"] == "TRADE"]
    needle = search.strip().lower()
    if needle:
        output = [row for row in output if needle in row["message"].lower()]
    return output


def csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["timestamp_ist", "level", "message"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def log_box(rows: list[dict[str, str]]) -> str:
    css = {"ERROR": "err", "WARNING": "warn", "TRADE": "trade", "INFO": "info"}
    lines = []
    for row in rows:
        text = html.escape(f'{row["timestamp_ist"]}  [{row["level"]}]  {row["message"]}')
        lines.append(f'<span class="{css[row["level"]]}">{text}</span>')
    return '<div class="logbox">' + "<br>".join(lines or ["No matching log lines."]) + "</div>"


st.title("📈 Nifty Scalper Operations Console")
st.caption("Read-only monitoring • live logs • structured exports • mobile compatible")

with st.sidebar:
    st.header("Live log controls")
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_seconds = st.select_slider("Refresh every", options=[3, 5, 10, 15, 30], value=5)
    line_limit = st.select_slider("Recent lines", options=[200, 500, 1000, 2000, 5000], value=1000)
    level_filter = st.selectbox("Severity", ["ALL", "INFO", "WARNING", "ERROR", "TRADE"])
    trades_only = st.toggle("Trade events only", value=False)
    search_text = st.text_input("Contains text", placeholder="ORDER_SENT, ERROR, symbol…")
    if st.button("Refresh now", use_container_width=True):
        st.rerun()

livez = api_json("/livez")
readyz = api_json("/readyz")
trading = api_json("/health/trading")
broker = (trading or {}).get("broker") or {}

cols = st.columns(6)
cols[0].metric("API", "ONLINE" if livez else "OFFLINE")
cols[1].metric("Bot loaded", "YES" if (livez or {}).get("bot_loaded") else "NO")
cols[2].metric("Execution", "READY" if (trading or {}).get("ready") else "BLOCKED")
cols[3].metric("Live orders", "ARMED" if (trading or {}).get("live_orders_armed") else "OFF")
cols[4].metric("Broker", "READY" if broker.get("ready") else "NOT READY")
cols[5].metric("Balance", str(broker.get("balance") or "—"))

blockers = (trading or {}).get("blockers") or (readyz or {}).get("blockers") or []
if blockers:
    st.warning("Current blockers: " + ", ".join(map(str, blockers)))
elif livez:
    st.success("No operational blockers reported.")
else:
    st.error("Bot API is unreachable from the dashboard.")


def render_live() -> None:
    text, error = journal_text(lines=line_limit)
    if error:
        st.error(f"Unable to read {SERVICE_NAME} journal: {error}")
        return
    rows = filtered(parse_logs(text), level_filter, search_text, trades_only)
    stats = st.columns(4)
    stats[0].metric("Visible rows", len(rows))
    stats[1].metric("Errors", sum(row["level"] == "ERROR" for row in rows))
    stats[2].metric("Warnings", sum(row["level"] == "WARNING" for row in rows))
    stats[3].metric("Trade events", sum(row["level"] == "TRADE" for row in rows))
    st.markdown(log_box(rows), unsafe_allow_html=True)
    st.caption(f"Updated {datetime.now().astimezone().strftime('%d %b %Y %I:%M:%S %p %Z')}")


st.subheader("Live logs")
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_seconds}s")
    def live_fragment() -> None:
        render_live()
    live_fragment()
else:
    render_live()

st.divider()
st.subheader("Download logs by date and exact time")
export_cols = st.columns([1.2, 1, 1, 1, 1.4])
export_date = export_cols[0].date_input("Date", value=date.today(), max_value=date.today())
start_at = export_cols[1].time_input("From", value=time(9, 15), step=60)
end_at = export_cols[2].time_input("To", value=time(15, 30), step=60)
export_level = export_cols[3].selectbox("Level", ["ALL", "INFO", "WARNING", "ERROR", "TRADE"], key="export_level")
export_search = export_cols[4].text_input("Contains", key="export_search", placeholder="optional")

if start_at >= end_at:
    st.error("The From time must be earlier than the To time.")
else:
    since = f"{export_date.isoformat()} {start_at.strftime('%H:%M:%S')}"
    until = f"{export_date.isoformat()} {end_at.strftime('%H:%M:%S')}"
    text, error = journal_text(lines=20000, since=since, until=until)
    rows = filtered(parse_logs(text), export_level, export_search) if not error else []
    if error:
        st.error(f"Export query failed: {error}")
    else:
        table_tab, preview_tab = st.tabs(["Structured table", "Log preview"])
        with table_tab:
            st.dataframe(pd.DataFrame(rows, columns=["timestamp_ist", "level", "message"]), use_container_width=True, hide_index=True, height=320)
        with preview_tab:
            st.markdown(log_box(rows[-500:]), unsafe_allow_html=True)
        filename = f"niftybot-logs-{export_date.isoformat()}-{start_at.strftime('%H%M')}-{end_at.strftime('%H%M')}.csv"
        st.download_button(f"⬇️ Download CSV — {len(rows):,} rows", data=csv_bytes(rows), file_name=filename, mime="text/csv", type="primary", use_container_width=True)

with st.expander("Detailed bot diagnostics"):
    st.json({"livez": livez, "readyz": readyz, "trading": trading})

st.caption("This console is read-only. It cannot place, modify, or cancel broker orders.")
